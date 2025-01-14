#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int array[9];
    int reversedArray[9];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter 9 integers: ");
        for (int i = 0; i < 9; i++) {
            scanf("%d", &array[i]);
        }
    }

    MPI_Bcast(array, 9, MPI_INT, 0, MPI_COMM_WORLD);

    reversedArray[8 - rank] = array[rank];  // Process rank i places array[i] at reversedArray[8-i]

    MPI_Gather(&reversedArray[8 - rank], 1, MPI_INT, reversedArray, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Reversed Array: ");
        for (int i = 0; i < 9; i++) {
            printf("%d ", reversedArray[i]);
        }
        printf("\n");
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
