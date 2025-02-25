#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel function to concatenate the input string N times in parallel
__global__ void concatenate_kernel(const char* Sin, char* Sout, int Sin_len, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    
    // Each thread is responsible for copying the corresponding character from Sin to Sout
    if (idx < Sin_len * N) {
        int char_index = idx % Sin_len; // Calculate corresponding character index in Sin
        Sout[idx] = Sin[char_index];    // Copy character from Sin to Sout
    }
}

int main() {
    // Input string and number of repetitions
    char Sin[128];
    int N;

    printf("Enter the input string (Sin): ");
    fgets(Sin, sizeof(Sin), stdin);
    
    // Remove the newline character added by fgets
    Sin[strcspn(Sin, "\n")] = '\0';

    printf("Enter the number of repetitions (N): ");
    scanf("%d", &N);

    // Length of the input string
    int Sin_len = strlen(Sin);

    // Allocate memory for device variables
    char* d_Sin;
    char* d_Sout;

    // Output string will have length Sin_len * N
    int Sout_len = Sin_len * N;

    cudaMalloc((void**)&d_Sin, Sin_len * sizeof(char));
    cudaMalloc((void**)&d_Sout, Sout_len * sizeof(char));

    // Copy input string to device
    cudaMemcpy(d_Sin, Sin, Sin_len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel with one block and sufficient threads to cover all characters in Sout
    int blockSize = 256;
    int numBlocks = (Sout_len + blockSize - 1) / blockSize; // Calculate number of blocks
    concatenate_kernel<<<numBlocks, blockSize>>>(d_Sin, d_Sout, Sin_len, N);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Allocate space for the output string on the host
    char* Sout = (char*)malloc(Sout_len + 1);
    Sout[Sout_len] = '\0'; // Null-terminate the string

    // Copy the output string from device to host
    cudaMemcpy(Sout, d_Sout, Sout_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Output string Sout: %s\n", Sout);

    // Free allocated memory
    cudaFree(d_Sin);
    cudaFree(d_Sout);
    free(Sout);

    return 0;
}
