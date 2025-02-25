#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel function to generate the output string T from input string Sin
__global__ void generate_output_string_kernel(const char* Sin, char* T, int Sin_len, int* output_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    if (idx < Sin_len) {
        // Number of repetitions of current character
        int repetitions = idx + 1;
        
        // Calculate the starting index of the character in T
        int start_index = atomicAdd(output_index, repetitions);  // Atomic operation to ensure correct indexing

        // Store the character in the output string T at the correct positions
        for (int i = 0; i < repetitions; i++) {
            T[start_index + i] = Sin[idx];
        }
    }
}

int main() {
    // Input string Sin
    char Sin[128];  // Assuming the input string won't exceed 128 characters

    printf("Enter the input string Sin: ");
    fgets(Sin, sizeof(Sin), stdin);

    // Remove the trailing newline character if present
    Sin[strcspn(Sin, "\n")] = '\0';

    // Length of the input string Sin
    int Sin_len = strlen(Sin);
    
    // Length of output string T (triangular number sum)
    int T_len = (Sin_len * (Sin_len + 1)) / 2;  // Length of output string T

    // Allocate memory for device variables
    char* d_Sin;
    char* d_T;
    int* d_output_index;

    cudaMalloc((void**)&d_Sin, Sin_len * sizeof(char));
    cudaMalloc((void**)&d_T, T_len * sizeof(char));
    cudaMalloc((void**)&d_output_index, sizeof(int));

    // Initialize output index to 0
    int h_output_index = 0;
    cudaMemcpy(d_output_index, &h_output_index, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Sin, Sin, Sin_len * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (Sin_len + blockSize - 1) / blockSize;  // Calculate the number of blocks

    // Launch kernel
    generate_output_string_kernel<<<numBlocks, blockSize>>>(d_Sin, d_T, Sin_len, d_output_index);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Allocate space for the output string on the host
    char* T = (char*)malloc(T_len + 1);  // +1 for null-terminator
    T[T_len] = '\0';  // Null-terminate the output string

    // Copy the output string from device to host
    cudaMemcpy(T, d_T, T_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Output string T: %s\n", T);

    // Free allocated memory
    cudaFree(d_Sin);
    cudaFree(d_T);
    cudaFree(d_output_index);
    free(T);

    return 0;
}
