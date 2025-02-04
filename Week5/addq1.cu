#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel function to perform the operation y = a * x + y
__global__ void linearAlgebraKernel(float *x, float *y, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index

    if (idx < N) {
        y[idx] = a * x[idx] + y[idx];  // Perform the operation y = a * x + y
    }
}

void linearAlgebra(float *x, float *y, float a, int N) {
    float *d_x, *d_y;
    size_t size = N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks required
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Round up

    // Launch kernel to perform the operation
    linearAlgebraKernel<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, a, N);

    // Copy the result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int N = 10;  // Length of the vectors
    float a = 2.0f;  // Scalar value
    float x[N], y[N];

    // Initialize input vector x and y (for example purposes)
    for (int i = 0; i < N; i++) {
        x[i] = i + 1.0f;  // x = [1, 2, 3, ..., 10]
        y[i] = (i + 1) * 2.0f;  // y = [2, 4, 6, ..., 20]
    }

    // Perform the linear algebra operation y = ax + y
    linearAlgebra(x, y, a, N);

    // Print the result
    printf("Resulting vector y after the operation y = ax + y:\n");
    for (int i = 0; i < N; i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }

    return 0;
}
