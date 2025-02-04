#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256  // Define the number of threads per block

// Kernel function to perform selection sort on each row of the matrix
__global__ void selectionSortRows(float *matrix, int numCols, int numRows) {
    int row = blockIdx.x;  // Each block is responsible for one row
    int threadIdxInRow = threadIdx.x;  // Thread ID within the row

    if (row < numRows) {
        // Only process the row if the row index is valid
        for (int i = 0; i < numCols - 1; i++) {
            if (threadIdxInRow == i) {
                int minIdx = i;
                // Find the minimum element in the unsorted part of the row
                for (int j = i + 1; j < numCols; j++) {
                    if (matrix[row * numCols + j] < matrix[row * numCols + minIdx]) {
                        minIdx = j;
                    }    // Print the matrix before sorting
                }

                // Swap the minimum element with the current element
                if (minIdx != i) {
                    float temp = matrix[row * numCols + i];
                    matrix[row * numCols + i] = matrix[row * numCols + minIdx];
                    matrix[row * numCols + minIdx] = temp;
                }
            }
            __syncthreads();  // Synchronize threads to avoid race conditions
        }
    }
}

void sortMatrixRows(float *matrix, int numCols, int numRows) {
    float *d_matrix;
    size_t size = numRows * numCols * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void**)&d_matrix, size);

    // Copy matrix data from host to device
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    // Launch kernel: one block per row
    selectionSortRows<<<numRows, THREADS_PER_BLOCK>>>(d_matrix, numCols, numRows);

    // Copy the sorted matrix back to host
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
}

int main() {
    int numRows = 3;  // Number of rows in the matrix
    int numCols = 5;  // Number of columns in the matrix

    // Allocate memory for the matrix on the host
    float matrix[numRows][numCols] = {
        {5.4, 2.3, 9.1, 7.6, 1.2},
        {3.7, 8.5, 4.6, 2.1, 6.4},
        {9.0, 3.5, 7.2, 1.1, 8.7}
    };

    printf("Matrix before sorting each row:\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }

    // Sort the rows of the matrix
    sortMatrixRows((float*)matrix, numCols, numRows);

    printf("\nMatrix after sorting each row:\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
