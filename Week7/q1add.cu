#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_WORD_LENGTH 100
#define MAX_WORDS 100

__global__ void reverseWordsKernel(char* d_words, int* d_word_lengths, int num_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_words) {
        int length = d_word_lengths[idx];
        char* word = &d_words[idx * MAX_WORD_LENGTH];

        for (int i = 0; i < length / 2; i++) {
            char temp = word[i];
            word[i] = word[length - 1 - i];
            word[length - 1 - i] = temp;
        }
    }
}

void reverseWords(char* h_input) {
    char* h_words[MAX_WORDS];
    int h_word_lengths[MAX_WORDS];
    int num_words = 0;

    char* token = strtok(h_input, " ");
    while (token != NULL && num_words < MAX_WORDS) {
        h_words[num_words] = token;
        h_word_lengths[num_words] = strlen(token);
        num_words++;
        token = strtok(NULL, " ");
    }

    char* d_words;
    int* d_word_lengths;
    cudaMalloc((void**)&d_words, MAX_WORDS * MAX_WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_word_lengths, MAX_WORDS * sizeof(int));

    for (int i = 0; i < num_words; i++) {
        cudaMemcpy(&d_words[i * MAX_WORD_LENGTH], h_words[i], h_word_lengths[i] * sizeof(char), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_word_lengths, h_word_lengths, num_words * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
    int blocksPerGrid = (num_words + threadsPerBlock - 1) / threadsPerBlock;
    reverseWordsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_words, d_word_lengths, num_words);

    for (int i = 0; i < num_words; i++) {
        cudaMemcpy(h_words[i], &d_words[i * MAX_WORD_LENGTH], h_word_lengths[i] * sizeof(char), cudaMemcpyDeviceToHost);
    }

    printf("Reversed words:\n");
    for (int i = 0; i < num_words; i++) {
        printf("%s ", h_words[i]);
    }
    printf("\n");

    cudaFree(d_words);
    cudaFree(d_word_lengths);
}

int main() {
    char input[MAX_WORDS * MAX_WORD_LENGTH];

    printf("Enter a string: ");
    fgets(input, sizeof(input), stdin);
    
    size_t len = strlen(input);
    if (len > 0 && input[len - 1] == '\n') {
        input[len - 1] = '\0';
    }

    printf("Input: %s\n", input);
    reverseWords(input);
    return 0;
}