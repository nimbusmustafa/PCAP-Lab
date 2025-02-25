#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  // For malloc and free

__global__ void selectionSortKernel(int *arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - 1) {
        int minIdx = tid;
        
        // Perform selection sort on the element at index 'tid'
        for (int i = tid + 1; i < n; i++) {
            if (arr[i] < arr[minIdx]) {
                minIdx = i;
            }
        }

        // Swap the found minimum element with the element at 'tid'
        if (minIdx != tid) {
            int temp = arr[tid];
            arr[tid] = arr[minIdx];
            arr[minIdx] = temp;
        }

        // Ensure all threads are done before the next phase
        __syncthreads();
    }
}

int main() {
    int *arr, n;
    printf("Enter the number of elements in the array:\n");
    scanf("%d", &n);
    
    arr = (int *)malloc(n * sizeof(int));
    printf("Enter the elements:\n");
    for (int i = 0; i < n; ++i) {
        scanf("%d", &arr[i]);
    }

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void **)&d_arr, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set thread and block limits
    int threadLimit = 256;  // You can experiment with the number of threads per block
    int blockLimit = (n + threadLimit - 1) / threadLimit;  // Number of blocks needed

    // Launch kernel for selection sort
    for (int i = 0; i < n - 1; i++) {
        selectionSortKernel<<<blockLimit, threadLimit>>>(d_arr, n);
        // Wait for the GPU to finish before the next iteration
        cudaDeviceSynchronize();
    }

    // Copy the sorted array back to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_arr);
    free(arr);

    return 0;
}
