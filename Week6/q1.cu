#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolution_1d(const float* N, const float* M, float* P, int width, int mask_width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width) {
        float result = 0.0f;

        for (int k = 0; k < mask_width; ++k) {
            if (idx + k < width) {
                result += N[idx + k] * M[k];
            }
        }

        P[idx] = result;
    }
}

int main() {
    int width = 12;          
    int mask_width = 5;      
    float *N = (float*)malloc(width * sizeof(float));
    float *M = (float*)malloc(mask_width * sizeof(float));
    float *P = (float*)malloc(width * sizeof(float));

    for (int i = 0; i < width; i++) {
        N[i] = (float)i;  
    }


    for (int i = 0; i < mask_width; i++) {
        M[i] = 5;  
    }

    float *d_N, *d_M, *d_P;

    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, mask_width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;  
    int numBlocks = (width + blockSize - 1) / blockSize;  
    convolution_1d<<<numBlocks, blockSize>>>(d_N, d_M, d_P, width, mask_width);

    cudaDeviceSynchronize();

    cudaMemcpy(P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result of 1D Convolution:\n");
    for (int i = 0; i < width; i++) {
        printf("P[%d] = %.3f\n", i, P[i]);
    }

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    free(N);
    free(M);
    free(P);

    return 0;
}