#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel function to construct decreasing substrings of the input string S
__global__ void print_decreasing_substrings_kernel(const char* S, int S_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    // Each thread constructs and prints a substring of decreasing lengths
    if (idx < S_len) {
        // Calculate the substring length
        int substring_len = S_len - idx;

        // Create a temporary buffer to store the substring
        char substring[128];  // Assuming S_len <= 128 for simplicity

        // Copy the first `substring_len` characters from S to the substring buffer
        for (int i = 0; i < substring_len; i++) {
            substring[i] = S[i];
        }
        substring[substring_len] = '\0';  // Null-terminate the substring

        // Print the substring
        printf("%s", substring);  // Print the result for the current thread
    }
}

int main() {
    // Input string S
    char S[128];  // Assuming the input string length won't exceed 128 characters

    printf("Enter the input string S: ");
    fgets(S, sizeof(S), stdin);

    // Remove the trailing newline character (if present)
    S[strcspn(S, "\n")] = '\0';

    // Length of the input string S
    int S_len = strlen(S);

    // Allocate memory on the device for the input string
    char* d_S;
    cudaMalloc((void**)&d_S, S_len * sizeof(char));

    // Copy input string to device
    cudaMemcpy(d_S, S, S_len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel to print decreasing substrings
    int blockSize = 256;
    int numBlocks = (S_len + blockSize - 1) / blockSize;  // Calculate the number of blocks

    print_decreasing_substrings_kernel<<<numBlocks, blockSize>>>(d_S, S_len);

    // Check for kernel errors
    cudaDeviceSynchronize();

    // Free allocated memory
    cudaFree(d_S);

    return 0;
}
