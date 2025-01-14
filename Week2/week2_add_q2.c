#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int factorial(int n) {
    if (n <= 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size, N, local_sum = 0;
    int result = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least two processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        printf("Enter the value of N: ");
        scanf("%d", &N);

        // Broadcast the value of N to all other processes
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // Non-root processes receive N
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Each process will handle a specific part of the series:
    // The first process handles 1!, the second handles (1+2), and so on.
    for (int i = rank; i < N; i += size) {
        if (i % 2 == 0) {
            // Even index (0, 2, 4, ...): compute factorial
            local_sum += factorial(i + 1);
        } else {
            // Odd index (1, 3, 5, ...): compute sum
            local_sum += (i + 1) * (i + 2) / 2;
        }
    }

    // Gather results from all processes to the root process
    MPI_Reduce(&local_sum, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The result of the series is: %d\n", result);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
