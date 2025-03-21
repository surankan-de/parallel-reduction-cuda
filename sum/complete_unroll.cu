#include <iostream>
#include<cuda_runtime.h>
#include <chrono>
#include <numeric> 

//full unroll
template <unsigned int blockSize>
__device__ void unroll_last(volatile int* sdata, int tid){
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void complete_unroll(int *g_in_data, int *g_out_data){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i+blockDim.x];
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] +=sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] +=sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] +=sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) unroll_last<blockSize>(sdata, tid);

    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }
}


int main(){
    int dataSize = 1 << 22;
    size_t byteSize = dataSize * sizeof(int);

    int *hostInput = new int[dataSize];
    int *hostOutput = new int[(dataSize + 255) / 256];

    int *deviceInput, *deviceOutput;

    srand(42);
    for (int i = 0; i < dataSize; i++) {
        hostInput[i] = rand() % 100;
    }

    cudaMalloc(&deviceInput, byteSize);
    cudaMalloc(&deviceOutput, (dataSize + 255) / 256 * sizeof(int));

    cudaMemcpy(deviceInput, hostInput, byteSize, cudaMemcpyHostToDevice);

    int threadCount = 256;
    int blockCount = (dataSize + (2 * threadCount) - 1) / (2 * threadCount);

    auto startTime = std::chrono::high_resolution_clock::now();

    switch (threadCount) {
        case 512:
            complete_unroll<512><<<blockCount, 512, 512 * sizeof(int)>>>(deviceInput, deviceOutput);
            break;
        case 256:
            complete_unroll<256><<<blockCount, 256, 256 * sizeof(int)>>>(deviceInput, deviceOutput);
            break;
        case 128:
            complete_unroll<128><<<blockCount, 128, 128 * sizeof(int)>>>(deviceInput, deviceOutput);
            break;
    }

    cudaDeviceSynchronize();

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;

    cudaMemcpy(hostOutput, deviceOutput, (dataSize + 255) / 256 * sizeof(int), cudaMemcpyDeviceToHost);

    int sumGPU = hostOutput[0];
    for (int i = 1; i < (dataSize + 255) / 256; ++i) {
        sumGPU += hostOutput[i];
    }

    int sumCPU = std::accumulate(hostInput, hostInput + dataSize, 0);
    if (sumCPU == sumGPU) {
        std::cout << "\033[32m";
        std::cout << "Verification successful: GPU result matches CPU result.\n";
        std::cout << "GPU Result: " << sumGPU << ", CPU Result: " << sumCPU << std::endl;
    } else {
        std::cout << "\033[31m";
        std::cout << "Verification failed: GPU result (" << sumGPU << ") does not match CPU result (" << sumCPU << ").\n";
        std::cout << "GPU Result: " << sumGPU << ", CPU Result: " << sumCPU << std::endl;
    }
    std::cout << "\033[0m";

    double memoryBandwidth = (elapsedTime > 0) ? (byteSize / elapsedTime / 1e6) : 0;
    std::cout << "Reduced result: " << sumGPU << std::endl;
    std::cout << "Time elapsed: " << elapsedTime << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << memoryBandwidth << " GB/s" << std::endl;

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutput;

}