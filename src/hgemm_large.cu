#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>

#include "common.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 8   // BLOCK_COLS / WMMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / WMMA_M

#define WARP_ROW_TILES 4  // WARP_COLS / WMMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / WMMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / WMMA_K

#define CHUNK_LINE_BYTES 64          // CHUNK_K * WMMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * WMMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

using namespace nvcuda;

__global__ void wmmaBaseKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, WMMA_M);
    const size_t N_tiles = div_ceil(N, WMMA_N);
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    half *smem_warp_tile_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS +
                               (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    half *smem_warp_stream_ptr = &smem[0][0] + warp_id * WMMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * WMMA_M * N + block_tile_j * WMMA_N;
    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * WMMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * WMMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            *((int4 *)&smem[A_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            *((int4 *)&smem[B_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_COL_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[WARP_ROW_TILES];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
                const half *A_tile_ptr = &smem[A_smem_idx][k_step * WMMA_K];

                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, WMMA_K * CHUNK_K);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
                const half *B_tile_ptr = &smem[B_smem_idx][k_step * WMMA_K];

                wmma::load_matrix_sync(B_frag[j], B_tile_ptr, WMMA_K * CHUNK_K);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    wmma::mma_sync(C_frag[i][j_s], A_frag[i], B_frag[j_s], C_frag[i][j_s]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *C_tile_ptr = smem_warp_tile_ptr + i * C_SMEM_STRIDE * WMMA_M + j * WMMA_N;

            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WMMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) + lane_id % 16);
    }
}

size_t initWmmaBase() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size =
        std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half), BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(wmmaBaseKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

bool read_matrices_from_dir(const std::string& dir,
                           std::vector<__half>& A_fp16,
                           std::vector<__half>& B_fp16,
                           int& M, int& N, int& K) {
    std::string path_A = dir + "/A_matrix.bin";
    std::string path_B = dir + "/B_matrix.bin";

    std::ifstream fa(path_A, std::ios::binary);
    std::ifstream fb(path_B, std::ios::binary);
    if (!fa.is_open() || !fb.is_open()) {
        std::cerr << "Error opening binary matrix files in " << dir << std::endl;
        return false;
    }

    int m_a = 0, k_a = 0, k_b = 0, n_b = 0;

    fa.read(reinterpret_cast<char*>(&m_a), sizeof(int));
    fa.read(reinterpret_cast<char*>(&k_a), sizeof(int));
    size_t size_A = static_cast<size_t>(m_a) * k_a;
    A_fp16.resize(size_A);
    fa.read(reinterpret_cast<char*>(A_fp16.data()), size_A * sizeof(__half));

    fb.read(reinterpret_cast<char*>(&k_b), sizeof(int));
    fb.read(reinterpret_cast<char*>(&n_b), sizeof(int));
    size_t size_B = static_cast<size_t>(k_b) * n_b;
    B_fp16.resize(size_B);
    fb.read(reinterpret_cast<char*>(B_fp16.data()), size_B * sizeof(__half));

    fa.close();
    fb.close();

    if (k_a != k_b) {
        std::cerr << "Error: K dimension mismatch between A and B\n";
        return false;
    }

    M = m_a;
    K = k_a;
    N = n_b;
    return true;
}


int main(int argc, char* argv[]) {
    std::string input_dir = "data/input/Case1_768x768x768";
    std::string output_dir = "data/output";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--indir") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--outdir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-d input_dir] [-o output_dir]" << std::endl;
            return 1;
        }
    }

    std::string case_name = input_dir.substr(input_dir.find_last_of("/\\") + 1);
    std::string output_file = output_dir + "/result_" + case_name + ".txt";

    // Read input matrices
    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;
    if (!read_matrices_from_dir(input_dir, A_fp16, B_fp16, M, N, K)) {
        return 1;
    }

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Async memory transfers
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_A, A_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, B_fp16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice, stream2);
    
    // Wait for transfers to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    double memcpy_time = std::chrono::duration<double, std::milli>(end_memcpy - start_memcpy).count();
    std::cout << "Async memcpy time: " << memcpy_time << " ms" << std::endl;

    static size_t smem_max_size = initWmmaBase();

    // Configure kernel launch
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    // Warmup run
    wmmaBaseKernel<<<grid, block, smem_max_size, stream3>>>(d_A, d_B, d_C, M, N, K);
    cudaStreamSynchronize(stream3);

    // Timing run
    auto start_kernel = std::chrono::high_resolution_clock::now();
    wmmaBaseKernel<<<grid, block, smem_max_size, stream3>>>(d_A, d_B, d_C, M, N, K);
    cudaStreamSynchronize(stream3);
    auto end_kernel = std::chrono::high_resolution_clock::now();

    double kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();

    // Copy results back
    std::vector<half> C_result(M * N);
    cudaMemcpyAsync(C_result.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);

    // Verify results
    float result_sum = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        result_sum += __half2float(C_result[i]);
    }

    // Calculate performance metrics
    double flops = 2.0 * M * N * K;
    double gflops = flops / (kernel_time / 1000.0) / 1e9;

    std::cout << "Optimized HGEMM Kernel Time: " << kernel_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Result sum: " << result_sum << std::endl;

    // Write results to file
    std::ofstream out(output_file);
    if (out.is_open()) {
        out << "Case: " << case_name << "\n";
        out << "Matrix dimensions: " << M << "x" << N << "x" << K << "\n";
        out << "Async memcpy time: " << memcpy_time << " ms\n";
        out << "Kernel execution time: " << kernel_time << " ms\n";
        out << "Performance: " << gflops << " GFLOPS\n";
        out << "Result sum: " << result_sum << "\n";
        out.close();
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}
