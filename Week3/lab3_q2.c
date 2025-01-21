#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, N, M;
    float avg = 0.0;
    int *A = NULL, *B = NULL;
    float *D = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        N = size;  // Total number of processes
        printf("Enter the value of M (number of elements per process):\n");
        scanf("%d", &M);

        // Dynamically allocate memory for array A to hold M * N elements
        A = (int *)malloc(M * N * sizeof(int));

        // Read M * N values from the user
        printf("Enter %d values for the array A:\n", M * N);
        for (int i = 0; i < M * N; i++) {
            scanf("%d", &A[i]);
        }
    }

    // Allocate memory for local arrays in each process
    B = (int *)malloc(M * sizeof(int));
    D = (float *)malloc(size * sizeof(float));  // One entry for each process

    // Broadcast the value of M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the data from root process to all other processes
    MPI_Scatter(A, M, MPI_INT, B, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the elements received by each process
    printf("Process %d received elements: ", rank);
    for (int i = 0; i < M; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    // Calculate the average of the M elements received by this process
    float local_sum = 0.0;
    for (int i = 0; i < M; i++) {
        local_sum += B[i];
    }
    avg = local_sum / M;

    // Print the average computed by each process
    printf("Process %d computed average: %f\n", rank, avg);

    // Gather the local averages from all processes
    MPI_Gather(&avg, 1, MPI_FLOAT, D, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // The root process computes the final average of all the averages
    if (rank == 0) {
        avg = 0.0;
        for (int i = 0; i < N; i++) {
            avg += D[i];
        }
        avg = avg / N;
        printf("\nFinal average of all processes = %f\n", avg);

        // Free dynamically allocated memory for array A
        free(A);
        free(D);
    }

    // Free dynamically allocated memory for local arrays in each process
    free(B);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
