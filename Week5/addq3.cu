#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256  // Number of threads per block

// Kernel function for Odd phase of Odd-Even Transposition Sort
__global__ void oddPhase(int *array, int n) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);  // Thread processes every second element

    if (idx + 1 < n && idx % 2 == 1) {
        // Compare and swap elements at indices idx and idx+1
        if (array[idx] > array[idx + 1]) {
            int temp = array[idx];
            array[idx] = array[idx + 1];
            array[idx + 1] = temp;
        }
    }
}

// Kernel function for Even phase of Odd-Even Transposition Sort
__global__ void evenPhase(int *array, int n) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);  // Thread processes every second element

    if (idx + 1 < n && idx % 2 == 0) {
        // Compare and swap elements at indices idx and idx+1
        if (array[idx] > array[idx + 1]) {
            int temp = array[idx];
            array[idx] = array[idx + 1];
            array[idx + 1] = temp;
        }
    }
}

// Function to perform Odd-Even Transposition Sort using CUDA
void oddEvenTranspositionSort(int *array, int n) {
    int *d_array;
    size_t size = n * sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void**)&d_array, size);

    // Copy input array from host to device
    cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks required for the odd and even phases
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Round up

    // Perform the odd-even transposition sort
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Odd phase
            oddPhase<<<blocks, THREADS_PER_BLOCK>>>(d_array, n);
        } else {
            // Even phase
            evenPhase<<<blocks, THREADS_PER_BLOCK>>>(d_array, n);
        }

        // Synchronize threads
        cudaDeviceSynchronize();
    }

    // Copy the sorted array back to the host
    cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
}

int main() {
    int n = 10;  // Size of the array
    int array[] = {9, 3, 7, 1, 5, 4, 2, 6, 8, 0};  // Example unsorted array

    // Print the array before sorting
    printf("Array before sorting:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    // Perform odd-even transposition sort
    oddEvenTranspositionSort(array, n);

    // Print the sorted array
    printf("\nArray after sorting:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    return 0;
}
