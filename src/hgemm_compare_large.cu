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

    // 通过交替行访问方向，可以减少全局内存访问的冲突，提高内存带宽利用率
    const size_t block_tile_i = (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    // 避免越界
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    // 动态共享内容，缓存数组
    extern __shared__ half smem[][AB_SMEM_STRIDE];

    // 线程id和线程内分工
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    // 共享内存中B矩阵的编译量
    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    // 线程数负责的warp指针，计算当前 warp 在共享内存中存储C矩阵 tile 的起始指针
    half *smem_warp_tile_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS + (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    // 数据流指针：
    // smem_warp_stream_ptr：当前 warp 在共享内存中用于数据流传输的起始指针。
    // gmem_idx：当前 warp 在全局内存中存储C矩阵结果的起始索引。
    // src_gmem_warp_stream_ptr：全局内存中C矩阵对应位置的指针。
    half *smem_warp_stream_ptr = &smem[0][0] + warp_id * WMMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * WMMA_M * N + block_tile_j * WMMA_N;
    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // 初始化WMMA累加片段
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // A, B的全局内存指针
    const half *A_warp_ptr = &A[block_tile_i * WMMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * WMMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    // 共享内存加载迭代次数
    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    // 沿矩阵乘法的 K 维度（最内层循环）分块处理，每次加载CHUNK_K个 tile 到共享内存，减少全局内存访问
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {

        // 加载 A 矩阵到共享内存
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

        // 加载 B 矩阵到共享内存
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

            // 加载A和B的WMMA片段
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

void hgemm_cublas_fp16(const __half* A_fp16, const __half* B_fp16, __half* C_fp16,
                       int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B_fp16, CUDA_R_16F, N,
                 A_fp16, CUDA_R_16F, K,
                 &beta,
                 C_fp16, CUDA_R_16F, N,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DFALT_TENSOR_OP);

    cublasDestroy(handle);
}

// 计算Frobenius范数的相对误差
__global__ void compute_fp16_relative_error_kernel(const __half* ref, const __half* calc,
                                                   float* diff_sq, float* ref_sq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ref_val = __half2float(ref[idx]);
        float calc_val = __half2float(calc[idx]);
        float diff = calc_val - ref_val;
        diff_sq[idx] = diff * diff;  // 计算差的平方
        ref_sq[idx] = ref_val * ref_val;  // 计算参考值的平方
    }
}

// GPU接口计算相对误差
float compute_relative_error_fp16_gpu(const __half* d_ref, const __half* d_calc, int size) {
    float *d_diff_sq, *d_ref_sq;
    cudaMalloc(&d_diff_sq, size * sizeof(float));
    cudaMalloc(&d_ref_sq, size * sizeof(float));

    int block = 256;
    int grid = (size + block - 1) / block;
    compute_fp16_relative_error_kernel<<<grid, block>>>(d_ref, d_calc, d_diff_sq, d_ref_sq, size);
    cudaDeviceSynchronize();

    std::vector<float> h_diff_sq(size);
    std::vector<float> h_ref_sq(size);
    cudaMemcpy(h_diff_sq.data(), d_diff_sq, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref_sq.data(), d_ref_sq, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_diff_sq);
    cudaFree(d_ref_sq);

    float sum_diff = 0.f, sum_ref = 0.f;
    for (int i = 0; i < size; ++i) {
        sum_diff += h_diff_sq[i];
        sum_ref += h_ref_sq[i];
    }

    return std::sqrt(sum_diff / sum_ref);
}

int main(int argc, char* argv[]) {
    std::string input_dir = "data/input/Case1_768x768x768";
    std::string output_dir = "data/output";

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

    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;
    if (!read_matrices_from_dir(input_dir, A_fp16, B_fp16, M, N, K)) return 1;

    __half *d_A_fp16, *d_B_fp16, *d_C_cublas, *d_C_custom;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_cublas, M * N * sizeof(__half));
    cudaMalloc(&d_C_custom, M * N * sizeof(__half));

    // 创建CUDA流
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // 创建事件
    cudaEvent_t event_A_done, event_B_done;
    cudaEventCreate(&event_A_done);
    cudaEventCreate(&event_B_done);

    // 异步拷贝A矩阵到设备，stream1
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_A_fp16, A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(event_A_done, stream1);  // 记录A拷贝完成事件

    // 异步拷贝B矩阵到设备，stream2
    cudaMemcpyAsync(d_B_fp16, B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(event_B_done, stream2);  // 记录B拷贝完成事件

    // 让计算流stream3等待A、B拷贝完成
    cudaStreamWaitEvent(stream3, event_A_done, 0);
    cudaStreamWaitEvent(stream3, event_B_done, 0);

    static size_t smem_max_size = initWmmaBase();

    // Configure kernel launch
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    wmmaBaseKernel<<<grid, block, smem_max_size, stream3>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
    cudaStreamSynchronize(stream3);

    auto end_memcpy = std::chrono::high_resolution_clock::now();
    double memcpy_time = std::chrono::duration<double, std::milli>(end_memcpy - start_memcpy).count();
    std::cout << "Async memcpy time: " << memcpy_time << " ms" << std::endl;

    // cublas计算保持同步调用（如果想也用stream可改）
    auto start = std::chrono::high_resolution_clock::now();
    hgemm_cublas_fp16(d_A_fp16, d_B_fp16, d_C_cublas, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double duration_cublas = std::chrono::duration<double, std::milli>(end - start).count();

    // 测试运行自定义核函数，异步调用放stream3
    auto start2 = std::chrono::high_resolution_clock::now();
    wmmaBaseKernel<<<grid, block, smem_max_size, stream3>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
    cudaStreamSynchronize(stream3);
    auto end2 = std::chrono::high_resolution_clock::now();
    double duration_custom = std::chrono::duration<double, std::milli>(end2 - start2).count();

    // 计算相对误差
    float rel_error = compute_relative_error_fp16_gpu(d_C_cublas, d_C_custom, M * N);

    // 拷贝结果回host（异步，用stream1）
    std::vector<__half> C_cublas_host(M * N);
    cudaMemcpy(C_cublas_host.data(), d_C_cublas, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<__half> C_custom_host(M * N);
    cudaMemcpyAsync(C_custom_host.data(), d_C_custom, M * N * sizeof(__half), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);

    float sum_cublas = 0.f, sum_custom = 0.f;
    for (int i = 0; i < M * N; ++i) {
        sum_cublas += __half2float(C_cublas_host[i]);
        sum_custom += __half2float(C_custom_host[i]);
    }

    double flops = 2.0 * M * N * K;
    double cublas_gflops = flops / (duration_cublas / 1000.0) / 1e9;
    double custom_gflops = flops / (duration_custom / 1000.0) / 1e9;

    float perf_ratio = static_cast<float>(custom_gflops / cublas_gflops * 100.0);
    perf_ratio = std::max(0.0f, std::min(100.0f, perf_ratio));
    float score = (rel_error > 0.05f || isnan(rel_error)) ? 0.0f : 70.0f * perf_ratio / 100.0f + 30.0f * (1.0f - rel_error);

    std::cout << "cuBLAS FP16 GEMM Time: " << duration_cublas << " ms, gFLOPS: " << cublas_gflops << std::endl;
    std::cout << "Custom FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << std::endl;
    std::cout << "Performance Ratio: " << perf_ratio << "%" << std::endl;
    std::cout << "Relative Error (GPU FP16 calc): " << rel_error << std::endl;
    std::cout << "Related Score: " << score << " / 100" << std::endl;
    std::cout << "cuBLAS Result sum: " << sum_cublas << std::endl;
    std::cout << "Custom Kernel Result sum: " << sum_custom << std::endl;

    std::ofstream out(output_file);
    if (out.is_open()) {
        out << "Case: " << case_name << "\n";
        out << "cuBLAS FP16 GEMM Time: " << duration_cublas << " ms, gFLOPS: " << cublas_gflops << "\n";
        out << "Custom FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << "\n";
        out << "Performance Ratio: " << perf_ratio << "\n";
        out << "Relative Error (GPU FP16 calc): " << rel_error << "\n";
        out << "Related Score: " << score << " / 100\n";
        out << "cuBLAS Result sum: " << sum_cublas << "\n";
        out << "Custom Kernel Result sum: " << sum_custom << "\n";
        out.close();
    }

    // 释放资源
    cudaEventDestroy(event_A_done);
    cudaEventDestroy(event_B_done);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_cublas);
    cudaFree(d_C_custom);

    return 0;
}