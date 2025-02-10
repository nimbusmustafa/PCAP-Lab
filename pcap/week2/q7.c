#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

#define MAX_WORD_LENGTH 100

void toggle_word(char *word) {
    int i = 0;
    while (word[i] != '\0') {
        if (isupper(word[i])) {
            word[i] = tolower(word[i]);
        } else if (islower(word[i])) {
            word[i] = toupper(word[i]);
        }
        i++;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    char word[MAX_WORD_LENGTH];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly two processes.\n");
            fflush(stdout);  // Ensure it prints immediately
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        // Sender process
        printf("Enter a word to send: ");
        fflush(stdout);  // 🔹 This forces immediate printing before scanf()
        
        scanf("%s", word);

        // Synchronous send to process 1
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent: %s\n", word);
        fflush(stdout);  // 🔹 Force print order

        // Synchronous receive from process 1
        MPI_Recv(word, MAX_WORD_LENGTH, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received back: %s\n", word);
        fflush(stdout);  // 🔹 Force print order
    } else if (rank == 1) {
        // Receiver process

        // Synchronous receive from process 0
        MPI_Recv(word, MAX_WORD_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received: %s\n", word);
        fflush(stdout);

        // Toggle the word
        toggle_word(word);
        printf("Process 1 toggled: %s\n", word);
        fflush(stdout);

        // Synchronous send back to process 0
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        printf("Process 1 sent back: %s\n", word);
        fflush(stdout);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
