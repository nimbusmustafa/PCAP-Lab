#include <iostream>
#include <cuda_runtime.h>

#define M 4  // Number of rows of A and C
#define N 5  // Number of columns of B and C
#define K 3  // Number of columns of A and rows of B

__global__ void matrixMultiplyRow(int *A, int *B, int *C, int K1, int N1) {
    int row = blockIdx.x;  // Each thread handles one row of C
    if (row < M) {
        for (int j = 0; j < N1; j++) {
            C[row * N1 + j] = 0;
            for (int k = 0; k < K1; k++) {
                C[row * N1 + j] += A[row * K1 + k] * B[k * N1 + j];
            }
        }
    }
}

int main() {
    int A[M][K] = {{1, 2, 3},
                   {4, 5, 6},
                   {7, 8, 9},
                   {10, 11, 12}};
    int B[K][N] = {{1, 2, 3, 4, 5},
                   {6, 7, 8, 9, 10},
                   {11, 12, 13, 14, 15}};
    int C[M][N];

    int *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one thread per row
    matrixMultiplyRow<<<M, 1>>>(d_A, d_B, d_C, K, N);

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
