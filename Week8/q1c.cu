#include <iostream>
#include <cuda_runtime.h>

#define M 4  // Number of rows
#define N 5  // Number of columns

__global__ void matrixAddElement(int *A, int *B, int *C, int M1 , int N1) {
    int row = blockIdx.x;
    int col = threadIdx.x; 
    if (row < M1 && col < N1) {
        C[row * N1 + col] = A[row * N1 + col] + B[row * N1 + col];
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

    // Launch kernel with one thread per element (grid size M, N; block size 1)
    dim3 blockDim(N, 1, 1);
    dim3 gridDim(M, 1, 1);
    matrixAddElement<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);

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
