#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define M 4  // Number of rows
#define N 5  // Number of columns

__global__ void matrixAddRow(int *A, int *B, int *C, int N1) {
    int row = blockIdx.x;  // Each thread handles one row
    if (row < M) {
        for (int j = 0; j < N1; j++) {
            C[row * N1 + j] = A[row * N1 + j] + B[row * N1 + j];
        }
    }
}

int main() {
    int A[M][N] = {{1, 2, 3, 4, 5},
                   {6, 7, 8, 9, 10},
                   {11, 12, 13, 14, 15},
                   {16, 17, 18, 19, 20}};
    int B[M][N] = {{21, 22, 23, 24, 25},
                   {26, 27, 28, 29, 30},
                   {31, 32, 33, 34, 35},
                   {36, 37, 38, 39, 40}};
    int C[M][N];

    int *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one thread per row (M rows)
    matrixAddRow<<<M, 1>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Resultant Matrix C:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
