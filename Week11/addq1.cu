#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_ITEMS 10
#define MAX_FRIENDS 100

__global__ void calculateTotal(float* prices, int* quantities, float* totals, int num_items) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < MAX_FRIENDS) {
        float total = 0.0f;
        for (int i = 0; i < num_items; ++i) {
            int q = quantities[idx * num_items + i];
            total += q * prices[i];
        }
        totals[idx] = total;
    }
}

int main() {
    const int num_items = 5;
    const char* items[] = {"Shoes", "Shirt", "Watch", "Bag", "Sunglasses"};
    float prices_host[MAX_ITEMS] = {50.0f, 30.0f, 100.0f, 80.0f, 60.0f};

    int N;
    printf("Enter number of friends: ");
    scanf("%d", &N);

    int quantities_host[MAX_FRIENDS * MAX_ITEMS] = {0};

    printf("\nItem Menu:\n");
    for (int i = 0; i < num_items; ++i) {
        printf("%d. %s - $%.2f\n", i, items[i], prices_host[i]);
    }

    for (int i = 0; i < N; ++i) {
        printf("\nFriend %d:\n", i + 1);
        for (int j = 0; j < num_items; ++j) {
            printf("  Quantity of %s: ", items[j]);
            scanf("%d", &quantities_host[i * num_items + j]);
        }
    }

    // Device memory
    float *prices_dev, *totals_dev;
    int *quantities_dev;

    cudaMalloc((void**)&prices_dev, num_items * sizeof(float));
    cudaMalloc((void**)&quantities_dev, N * num_items * sizeof(int));
    cudaMalloc((void**)&totals_dev, N * sizeof(float));

    cudaMemcpy(prices_dev, prices_host, num_items * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(quantities_dev, quantities_host, N * num_items * sizeof(int), cudaMemcpyHostToDevice);

    calculateTotal<<<(N + 255) / 256, 256>>>(prices_dev, quantities_dev, totals_dev, num_items);

    float totals_host[MAX_FRIENDS];
    cudaMemcpy(totals_host, totals_dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n--- Purchase Summary ---\n");
    for (int i = 0; i < N; ++i) {
        printf("Total purchase by Friend %d: $%.2f\n", i + 1, totals_host[i]);
    }

    // Cleanup
    cudaFree(prices_dev);
    cudaFree(quantities_dev);
    cudaFree(totals_dev);

    return 0;
}
