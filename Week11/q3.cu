#include <stdio.h>
#include <cuda.h>

#define N 1024  // Length of input array (should fit in a single block)

// CUDA kernel for inclusive scan using shared memory
__global__ void inclusiveScanKernel(int* input, int* output, int n) {
    __shared__ int temp[N];

    int tid = threadIdx.x;

    // Load input into shared memory
    if (tid < n)
        temp[tid] = input[tid];

    __syncthreads();

    // Inclusive scan (up-sweep phase)
    for (int offset = 1; offset < n; offset *= 2) {
        int t;
        if (tid >= offset)
            t = temp[tid - offset];
        __syncthreads();
        if (tid >= offset)
            temp[tid] += t;
        __syncthreads();
    }

    // Write result to output
    if (tid < n)
        output[tid] = temp[tid];
}


int main() {
    int h_input[N], h_output[N];
    int *d_input, *d_output;

    // Initialize input with random numbers
    for (int i = 0; i < N; ++i)
        h_input[i] = rand() % 10;

    // Show first 10 elements of input
    printf("Input:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_input[i]);
    printf("...\n");

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel (single block, N threads)
    inclusiveScanKernel<<<1, N>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display first 10 results
    printf("Inclusive scan result:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_output[i]);
    printf("...\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}