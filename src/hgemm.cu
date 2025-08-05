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

// Thread block size
constexpr int BLOCK_SIZE = 32;
// Double buffering
constexpr int BUFFER_COUNT = 2;

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

__global__ void hgemm_optimized(const half* __restrict__ A, 
                              const half* __restrict__ B,
                              half* __restrict__ C, 
                              int M, int N, int K) {
    // Double-buffered shared memory
    __shared__ half s_A[BUFFER_COUNT][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half s_B[BUFFER_COUNT][BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    
    // Accumulator registers
    float sum[2] = {0.0f, 0.0f};
    int write_idx = 0;
    int read_idx = 0;
    
    // Prefetch first tile
    int a_col = tx;
    int b_row = ty;
    if (row < M && a_col < K) {
        s_A[write_idx][ty][tx] = __ldg(&A[row * K + a_col]);
    } else {
        s_A[write_idx][ty][tx] = __float2half(0.0f);
    }
    
    if (b_row < K && col < N) {
        s_B[write_idx][ty][tx] = __ldg(&B[b_row * N + col]);
    } else {
        s_B[write_idx][ty][tx] = __float2half(0.0f);
    }
    
    __syncthreads();
    
    // Main computation loop
    for (int t = 1; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Prefetch next tile while computing current one
        write_idx = 1 - read_idx;
        
        a_col = t * BLOCK_SIZE + tx;
        b_row = t * BLOCK_SIZE + ty;
        
        // Asynchronous prefetch
        if (row < M && a_col < K) {
            s_A[write_idx][ty][tx] = __ldg(&A[row * K + a_col]);
        } else {
            s_A[write_idx][ty][tx] = __float2half(0.0f);
        }
        
        if (b_row < K && col < N) {
            s_B[write_idx][ty][tx] = __ldg(&B[b_row * N + col]);
        } else {
            s_B[write_idx][ty][tx] = __float2half(0.0f);
        }
        
        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            sum[0] += __half2float(s_A[read_idx][ty][k]) * __half2float(s_B[read_idx][k][tx]);
            sum[1] += __half2float(s_A[read_idx][ty][k+1]) * __half2float(s_B[read_idx][k+1][tx]);
            sum[0] += __half2float(s_A[read_idx][ty][k+2]) * __half2float(s_B[read_idx][k+2][tx]);
            sum[1] += __half2float(s_A[read_idx][ty][k+3]) * __half2float(s_B[read_idx][k+3][tx]);
        }
        
        __syncthreads();
        read_idx = write_idx;
    }
    
    // Process last tile
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += 4) {
        sum[0] += __half2float(s_A[read_idx][ty][k]) * __half2float(s_B[read_idx][k][tx]);
        sum[1] += __half2float(s_A[read_idx][ty][k+1]) * __half2float(s_B[read_idx][k+1][tx]);
        sum[0] += __half2float(s_A[read_idx][ty][k+2]) * __half2float(s_B[read_idx][k+2][tx]);
        sum[1] += __half2float(s_A[read_idx][ty][k+3]) * __half2float(s_B[read_idx][k+3][tx]);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum[0] + sum[1]);
    }
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

    // Configure kernel launch
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warmup run
    hgemm_optimized<<<grid, block, 0, stream3>>>(d_A, d_B, d_C, M, N, K);
    cudaStreamSynchronize(stream3);

    // Timing run
    auto start_kernel = std::chrono::high_resolution_clock::now();
    hgemm_optimized<<<grid, block, 0, stream3>>>(d_A, d_B, d_C, M, N, K);
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
