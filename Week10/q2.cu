#include <stdio.h>
#include <cuda.h>

#define DATA_SIZE 1024     // Length of input signal
#define KERNEL_SIZE 5      // Length of the convolution kernel

// Declare constant memory for the kernel
__constant__ float d_kernel[KERNEL_SIZE];

// CUDA Kernel for 1D Convolution
__global__ void convolve1D(float* input, float* output, int dataSize, int kernelSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int halfKernel = kernelSize / 2;

    if (i < dataSize) {
        float sum = 0.0f;
        for (int j = 0; j < kernelSize; ++j) {
            int inputIndex = i + j - halfKernel;

            // Handle boundary conditions (zero padding)
            if (inputIndex >= 0 && inputIndex < dataSize) {
                sum += input[inputIndex] * d_kernel[j];
            }
        }
        output[i] = sum;
    }
}

int main() {
    const int dataSize = DATA_SIZE;
    const int kernelSize = KERNEL_SIZE;
    const int dataBytes = dataSize * sizeof(float);
    const int kernelBytes = kernelSize * sizeof(float);

    // Host memory
    float h_input[dataSize], h_output[dataSize];
    float h_kernel[KERNEL_SIZE] = {0.2f, 0.5f, 1.0f, 0.5f, 0.2f};  // Example Gaussian-like kernel

    // Initialize input data
    for (int i = 0; i < dataSize; ++i) {
        h_input[i] = (float)(i % 10);  // Some pattern
    }

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataBytes);
    cudaMalloc(&d_output, dataBytes);

    // Copy input to device
    cudaMemcpy(d_input, h_input, dataBytes, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernelBytes);

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch convolution kernel
    convolve1D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, dataSize, kernelSize);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, dataBytes, cudaMemcpyDeviceToHost);

    // Display some results
    for (int i = 0; i < 10; ++i) {
        printf("Output[%d] = %f\n", i, h_output[i]);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
