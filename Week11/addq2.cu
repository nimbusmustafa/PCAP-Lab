#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define MASK_WIDTH 5

__constant__ float d_mask[MASK_WIDTH];

__global__ void convolution_1D_tiled(float* N, float* P, int width) {
    __shared__ float Ns[TILE_SIZE + MASK_WIDTH - 1];

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int halo = MASK_WIDTH / 2;
    int t = threadIdx.x;

    // Load center data
    if (n >= halo && n < (width - halo))
        Ns[t + halo] = N[n];
    else
        Ns[t + halo] = 0;

    // Load halo regions
    if (t < halo) {
        Ns[t] = (n >= halo) ? N[n - halo] : 0;

        int idx = n + blockDim.x;
        Ns[t + TILE_SIZE + halo] = (idx < width) ? N[idx] : 0;
    }

    __syncthreads();

    if (n < width) {
        float result = 0;
        for (int i = 0; i < MASK_WIDTH; i++)
            result += Ns[t + i] * d_mask[i];
        P[n] = result;
    }
}

int main() {
    const int width = 1024;
    float h_N[width], h_P[width];
    float h_mask[MASK_WIDTH] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

    for (int i = 0; i < width; ++i)
        h_N[i] = (float)i;

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, MASK_WIDTH * sizeof(float));

    int blocks = (width + TILE_SIZE - 1) / TILE_SIZE;
    convolution_1D_tiled<<<blocks, TILE_SIZE>>>(d_N, d_P, width);

    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolution Output (First 10 elements):\n");
    for (int i = 0; i < 10; ++i)
        printf("%.2f ", h_P[i]);
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
