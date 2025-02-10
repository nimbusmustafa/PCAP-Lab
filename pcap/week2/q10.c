#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;

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
        printf("Enter an integer value: ");
        scanf("%d", &value);

        // Send the value to process 1
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Root process (rank 0) sent value %d to process 1\n", value);

        // Receive the final incremented value from the last process
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value++;
        printf("Root process (rank 0) received value %d from process %d\n", value, size - 1);
    } else {
        // Other processes (process 1 to size-1)
        // Receive the value from the previous process
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value %d\n", rank, value);
        value++;

        // If not the last process, send the incremented value to the next process
        if (rank < size - 1) {
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            printf("Process %d sent incremented value %d to process %d\n", rank, value, rank + 1);
        } else {
            // Last process (size-1) sends the incremented value back to the root
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Process %d sent incremented value %d back to root (process 0)\n", rank, value);
        }
    }

    MPI_Finalize();
    return 0;
}