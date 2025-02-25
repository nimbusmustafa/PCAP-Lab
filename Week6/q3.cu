#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void evenPhase(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2;

    if (i < n - 1) {  
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

__global__ void oddPhase(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 2 + 1;

    if (i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

void oddEvenSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Perform the sorting for enough phases
    for (int phase = 0; phase < n; phase++) {
        evenPhase<<<gridSize, blockSize>>>(d_arr, n);
        cudaDeviceSynchronize();

        oddPhase<<<gridSize, blockSize>>>(d_arr, n);
        cudaDeviceSynchronize();
    }

    // Copy sorted array back to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    int n = 5;
    int arr[] = {12,1,67,32,45};

    printf("Initial array: ");
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");

    oddEvenSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");

    return 0;
}
