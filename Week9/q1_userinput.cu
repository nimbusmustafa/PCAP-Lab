#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for SpMV
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
    int rows, cols;
    cout << "Enter number of rows and columns: ";
    cin >> rows >> cols;

    // Input dense matrix
    vector<vector<float>> matrix(rows, vector<float>(cols));
    cout << "Enter matrix elements (row-wise):\n";
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            cin >> matrix[i][j];

    // Input vector
    vector<float> h_x(cols);
    cout << "Enter input vector of size " << cols << ":\n";
    for (int i = 0; i < cols; ++i)
        cin >> h_x[i];

    // Convert dense to CSR format
    vector<float> h_values;
    vector<int> h_colIndex;
    vector<int> h_rowPtr = {0};

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (matrix[i][j] != 0) {
                h_values.push_back(matrix[i][j]);
                h_colIndex.push_back(j);
            }
        }
        h_rowPtr.push_back(h_values.size());
    }

    int NNZ = h_values.size(); // Number of non-zeros
    vector<float> h_y(rows);   // Output vector

    // Device pointers
    float *d_values, *d_x, *d_y;
    int *d_colIndex, *d_rowPtr;

    // Allocate memory on GPU
    cudaMalloc(&d_values, NNZ * sizeof(float));
    cudaMalloc(&d_colIndex, NNZ * sizeof(int));
    cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_values, h_values.data(), NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndex, h_colIndex.data(), NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(rows, d_values, d_colIndex, d_rowPtr, d_x, d_y);

    // Copy result back
    cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Output result
    cout << "Result vector y = [ ";
    for (int i = 0; i < rows; ++i)
        cout << h_y[i] << " ";
    cout << "]\n";

    // Free GPU memory
    cudaFree(d_values);
    cudaFree(d_colIndex);
    cudaFree(d_rowPtr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
