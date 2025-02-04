#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAddKernel(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x;  // Each thread gets a unique index (0 to N-1)
    if (idx < N) {
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

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and N threads
    vectorAddKernel<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 10;
    int A[N], B[N], C[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Perform vector addition
    vectorAdd(A, B, C, N);

    // Print the result
    printf("Result Vector C:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    return 0;
}
