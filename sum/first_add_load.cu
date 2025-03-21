#include <iostream>
#include<cuda_runtime.h>
#include <chrono>
#include <numeric> 

//load first then start
__global__ void first_add_load(int *g_in_data, int *g_out_data){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i+blockDim.x];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        // check out the reverse loop above
        if (tid < s){   // then, we check tid to do our computation
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
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

auto startTime = std::chrono::high_resolution_clock::now();

int blockCount = (dataSize + (2 * threadCount) - 1) / (2 * threadCount);
cudaError_t error;
first_add_load<<<blockCount, threadCount, threadCount * sizeof(int)>>>(deviceInput, deviceOutput);
error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
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