#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Number of rows
#define NNZ 9  // Number of non-zero values

// CUDA kernel for SpMV using CSR format
__global__ void spmv_csr_kernel(int num_rows, const float *values, const int *colIndex, const int *rowPtr, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            dot += values[j] * x[colIndex[j]];
        }
        y[row] = dot;
    }
}

int main() {
    // Host CSR representation of a 4x4 sparse matrix:
    // [10  0  0  0]
    // [ 0 20  0 30]
    // [40  0 50 60]
    // [ 0  0  0 70]

    float h_values[NNZ]    = {10, 20, 30, 40, 50, 60, 70};
    int   h_colIndex[NNZ]  = {0, 1, 3, 0, 2, 3, 3};
    int   h_rowPtr[N + 1]  = {0, 1, 3, 6, 7};
    float h_x[4]           = {1, 2, 3, 4};  // Input vector
    float h_y[4];                           // Output vector

    // Device arrays
    float *d_values, *d_x, *d_y;
    int *d_colIndex, *d_rowPtr;

    // Allocate device memory
    cudaMalloc((void **)&d_values, NNZ * sizeof(float));
    cudaMalloc((void **)&d_colIndex, NNZ * sizeof(int));
    cudaMalloc((void **)&d_rowPtr, (N + 1) * sizeof(int));
    cudaMalloc((void **)&d_x, 4 * sizeof(float));
    cudaMalloc((void **)&d_y, 4 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_values, h_values, NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndex, h_colIndex, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, d_values, d_colIndex, d_rowPtr, d_x, d_y);

    // Copy result back
    cudaMemcpy(h_y, d_y, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result vector y = [");
    for (int i = 0; i < 4; i++) {
        printf(" %.2f", h_y[i]);
    }
    printf(" ]\n");

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_colIndex);
    cudaFree(d_rowPtr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
