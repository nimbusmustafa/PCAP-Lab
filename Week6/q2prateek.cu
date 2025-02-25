#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void find_min_index(int *arr, int *min_index, int start, int end) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= start && idx < end) {
        if (arr[idx] < arr[*min_index]) {
            atomicExch(min_index, idx); 
        }
    }
}

__global__ void swap_elements(int *arr, int i, int min_index) {
    int temp = arr[i];
    arr[i] = arr[min_index];
    arr[min_index] = temp;
}

void selection_sort_parallel(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;  
    int numBlocks = (n + blockSize - 1) / blockSize;  

    for (int i = 0; i < n - 1; i++) {
        int* d_min_index;
        int min_index = i;
        cudaMalloc((void**)&d_min_index, sizeof(int));
        cudaMemcpy(d_min_index, &min_index, sizeof(int), cudaMemcpyHostToDevice);

        find_min_index<<<numBlocks, blockSize>>>(d_arr, d_min_index, i + 1, n);

        cudaDeviceSynchronize();

        cudaMemcpy(&min_index, d_min_index, sizeof(int), cudaMemcpyDeviceToHost);

        if (min_index != i) {
            swap_elements<<<1, 1>>>(d_arr, i, min_index);
            cudaDeviceSynchronize();
        }

        cudaFree(d_min_index);
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int n;

    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *arr = (int *)malloc(n * sizeof(int));

    printf("Enter %d elements for the array: ", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    printf("Original Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    selection_sort_parallel(arr, n);

    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}