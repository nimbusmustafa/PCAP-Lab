#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel function to calculate sine of angles in radians
__global__ void computeSine(float *angles, float *sine_values, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (idx < N) {
        sine_values[idx] = sin(angles[idx]);  // Compute sine of the angle at index
    }
}

void computeSineValues(float *angles, float *sine_values, int N) {
    float *d_angles, *d_sine_values;
    size_t size = N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sine_values, size);

    // Copy input array to device
    cudaMemcpy(d_angles, angles, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks required
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Round up

    // Launch kernel to compute sine values
    computeSine<<<blocks, THREADS_PER_BLOCK>>>(d_angles, d_sine_values, N);

    // Copy the result back to host
    cudaMemcpy(sine_values, d_sine_values, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_sine_values);
}

int main() {
    int N = 10;  // Length of the array
    float angles[N], sine_values[N];

    // Initialize angles (in radians)
    for (int i = 0; i < N; i++) {
        angles[i] = i * 0.1f;  // Example angles from 0 to 0.9 radians
    }

    // Compute sine values for the angles
    computeSineValues(angles, sine_values, N);

    // Print the result
    printf("Angles (radians) and their sine values:\n");
    for (int i = 0; i < N; i++) {
        printf("Angle: %.3f, Sine: %.3f\n", angles[i], sine_values[i]);
    }

    return 0;
}
