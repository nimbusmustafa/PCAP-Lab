#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel to modify each row: raise elements to the power of (row_index + 1)
__global__ void row_power_kernel(float *matrix, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < M && col < N) {
        int idx = row * N + col;
        float val = matrix[idx];
        float power = row + 1;
        matrix[idx] = powf(val, power);
    }
}

int main() {
    int M, N;

    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int size = M * N * sizeof(float);

    // Allocate host memory
    float *h_matrix = (float *)malloc(size);

    // Input matrix
    printf("Enter matrix elements row-wise:\n");
    for (int i = 0; i < M * N; ++i) {
        scanf("%f", &h_matrix[i]);
    }

    // Allocate device memory
    float *d_matrix;
    cudaMalloc((void **)&d_matrix, size);

    // Copy to device
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // Kernel config
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    // Launch kernel
    row_power_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, M, N);

    // Copy result back
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Output result
    printf("Modified matrix:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%.2f ", h_matrix[i * N + j]);
        printf("\n");
    }

    // Cleanup
    free(h_matrix);
    cudaFree(d_matrix);

    return 0;
}
