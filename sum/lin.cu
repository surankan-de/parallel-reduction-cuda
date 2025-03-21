#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <numeric> 

__global__ void linear_sum(int *g_in_data, int *g_out_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(g_out_data, g_in_data[tid]); // Use atomic addition to sum elements
    }
}

int main() {
    int n = 1 << 22; // 4M elements
    size_t bytes = n * sizeof(int);

    // Host arrays
    int *host_input_data = new int[n];
    int host_output_data = 0;

    // Device arrays
    int *dev_input_data, *dev_output_data;
    cudaMalloc(&dev_input_data, bytes);
    cudaMalloc(&dev_output_data, sizeof(int));
    cudaMemset(dev_output_data, 0, sizeof(int)); // Initialize output on GPU

    // Initialize input data
    srand(42);
    for (int i = 0; i < n; i++) {
        host_input_data[i] = rand() % 100;
    }

    // Copy data to GPU
    cudaMemcpy(dev_input_data, host_input_data, bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int num_blocks = (n + blockSize - 1) / blockSize;
    
    auto start = std::chrono::high_resolution_clock::now();
    linear_sum<<<num_blocks, blockSize>>>(dev_input_data, dev_output_data, n);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

    // Copy result back to CPU
    cudaMemcpy(&host_output_data, dev_output_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    // CPU verification
    int cpuResult = std::accumulate(host_input_data, host_input_data + n, 0);
    if (cpuResult == host_output_data) {
        std::cout << "\033[32mVerification successful: GPU result matches CPU result.\n";
    } else {
        std::cout << "\033[31mVerification failed: GPU result does not match CPU result.\n";
    }
    std::cout << "GPU Result: " << host_output_data << ", CPU Result: " << cpuResult << "\033[0m\n";
    
    std::cout << "Time elapsed: " << duration << " ms\n";
    
    // Free memory
    cudaFree(dev_input_data);
    cudaFree(dev_output_data);
    delete[] host_input_data;
}
