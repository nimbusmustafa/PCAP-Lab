#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256  // Constant number of threads per block

// Kernel function to add two vectors
__global__ void vectorAddKernel(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate the global thread index

    if (idx < N) {  // Make sure the thread is within bounds
        C[idx] = A[idx] + B[idx];
    }
}

void vectorAdd(int *A, int *B, int *C, int N) {
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks required
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Round up

    // Launch kernel with calculated number of blocks and threads per block
    vectorAddKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1000;  // Length of the vectors
    int A[N], B[N], C[N];

    // Initialize input vectors A and B
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Perform vector addition on the device
    vectorAdd(A, B, C, N);

    // Print the result (only the first 10 elements for brevity)
    printf("Result Vector C:\n");
    for (int i = 0; i < 10; i++) {  // Print the first 10 elements
        printf("%d ", C[i]);
    }
    printf("\n");

    return 0;
}
