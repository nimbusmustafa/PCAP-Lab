#include <stdio.h>
#include <cuda.h>

#define N 1024  // Matrix size N x N
#define BLOCK_SIZE 16  // Block size (BLOCK_SIZE x BLOCK_SIZE)

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row of C to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column of C to compute

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void printMatrix(const char* name, float* mat, int n, int sampleSize) {
    printf("%s (top-left %dx%d):\n", name, sampleSize, sampleSize);
    for (int i = 0; i < sampleSize; ++i) {
        for (int j = 0; j < sampleSize; ++j) {
            printf("%.1f ", mat[i * n + j]);
        }
        printf("\n");
    }
    printf("...\n\n");
}

int main() {
    int size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Print part of the inputs
    printMatrix("Matrix A", h_A, N, 4);
    printMatrix("Matrix B", h_B, N, 4);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the matrix multiplication kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result
    printf("Matrix C (top-left 4x4):\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    printf("...\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
