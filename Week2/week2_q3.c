#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;  
    int *array = NULL;
    int value;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the total number of elements to N (equal to number of processes)
    N = size;

    if (rank == 0) {
        // Root process: Read N elements
        array = (int *)malloc(N * sizeof(int));
        printf("Enter %d elements:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &array[i]);
        }

        // Allocate a buffer for buffered send
        int buffer_size = N * sizeof(int) + MPI_BSEND_OVERHEAD;
        void *buffer = malloc(buffer_size);
        MPI_Buffer_attach(buffer, buffer_size);

        // Send one value to each slave process
        for (int i = 1; i < N; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Root process sent %d to process %d\n", array[i], i);
        }

        // Root process will compute its own value
        printf("Root process (rank 0) received its value: %d\n", array[0]);
        printf("Root process (rank 0) computes: %d^2 = %d\n", array[0], array[0] * array[0]);

        // Detach the buffer
        MPI_Buffer_detach(&buffer, &buffer_size);

        // Free the allocated memory for array
        free(array);
    } else {
        // Slave processes: Receive the value
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value %d\n", rank, value);

        if (rank % 2 == 0) {
            printf("Process %d computes: %d^2 = %d\n", rank, value, value * value);
        } else {
            printf("Process %d computes: %d^3 = %d\n", rank, value, value * value * value);
        }
    }

    MPI_Finalize();
    return 0;
}
