#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Device function to check if the word matches at a given index
__device__ bool check_word_match(const char* sentence, int sentence_length, const char* word, int word_length, int index) {
    if (index + word_length > sentence_length) return false;
    
    for (int i = 0; i < word_length; i++) {
        if (sentence[index + i] != word[i]) {
            return false;
        }
    }
    return true;
}

// Kernel to count occurrences of the word in the sentence
__global__ void count_word_kernel(const char* sentence, int sentence_length, const char* word, int word_length, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (idx >= sentence_length - word_length + 1) return; // Bounds check

    if (check_word_match(sentence, sentence_length, word, word_length, idx)) {
        atomicAdd(count, 1); // Atomically add to the count
    }
}

int main() {
    // Input sentence and word from user
    char sentence[1024];
    char word[100];

    printf("Enter the sentence: ");
    fgets(sentence, sizeof(sentence), stdin);

    // Remove the trailing newline character
    sentence[strcspn(sentence, "\n")] = '\0';

    printf("Enter the word to search for: ");
    fgets(word, sizeof(word), stdin);

    // Remove the trailing newline character
    word[strcspn(word, "\n")] = '\0';

    // Get the lengths of the sentence and word
    int sentence_length = strlen(sentence);
    int word_length = strlen(word);

    // Allocate memory for device variables
    char* d_sentence;
    char* d_word;
    int* d_count;

    cudaMalloc((void**)&d_sentence, sentence_length * sizeof(char));
    cudaMalloc((void**)&d_word, word_length * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_sentence, sentence, sentence_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, word_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int)); // Initialize the count to 0

    // Launch kernel with one block and multiple threads
    int blockSize = 256;
    int numBlocks = (sentence_length + blockSize - 1) / blockSize; // Calculate number of blocks
    count_word_kernel<<<numBlocks, blockSize>>>(d_sentence, sentence_length, d_word, word_length, d_count);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Retrieve the count from device to host
    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", word, count);

    // Free allocated memory
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
