#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there is at least one slave process
    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least one slave process (total processes >= 2).\n");
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        // Master process
        for (int i = 1; i < size; i++) {
            number = i * 10;  // Arbitrary number to send to each slave
            printf("Master (process %d) sending number %d to slave %d\n", rank, number, i);
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send number to each slave
        }
    } else {
        // Slave process
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive number from master
        printf("Slave (process %d) received number: %d\n", rank, number);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}