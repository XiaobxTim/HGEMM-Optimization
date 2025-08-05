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

// 线程块大小
constexpr int BLOCK_SIZE = 32;
// 双缓冲
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

__global__ void hgemm_kernel(const half* __restrict__ A, 
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

    // 预热运行：在stream3执行
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hgemm_kernel<<<grid, block, 0, stream3>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
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
    hgemm_kernel<<<grid, block, 0, stream3>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
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