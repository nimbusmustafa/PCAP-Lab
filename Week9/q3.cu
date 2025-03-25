#include <stdio.h>
#include <cuda_runtime.h>

__global__ void onesComplementKernel(int *A, int *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;

        if (row == 0 || row == M - 1 || col == 0 || col == N - 1) {
            B[idx] = A[idx];  // Border: keep same
        } else {
            int val = A[idx] & 0xF;        // Ensure 4-bit
            B[idx] = (~val) & 0xF;         // 4-bit 1's complement
        }
    }
}

// Convert int to 4-bit binary string
void to4BitBinary(int num, char *binStr) {
    for (int i = 3; i >= 0; i--) {
        binStr[3 - i] = ((num >> i) & 1) ? '1' : '0';
    }
    binStr[4] = '\0';
}

int main() {
    int M, N;
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int size = M * N * sizeof(int);
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);

    printf("Enter matrix elements in decimal (0–15):\n");
    for (int i = 0; i < M * N; i++)
        scanf("%d", &h_A[i]);

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    onesComplementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("\nOutput Matrix (decimal borders, 4-bit 1's complement inside):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (i == 0 || i == M - 1 || j == 0 || j == N - 1) {
                printf("%d\t", h_B[idx]);
            } else {
                char binStr[5];
                to4BitBinary(h_B[idx], binStr);
                printf("%s\t", binStr);
            }
        }
        printf("\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
