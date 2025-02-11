#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, mat[4][4], out[4], i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program must be run with 4 MPI processes.\n");
        }
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                scanf("%d", &mat[i][j]);
    }

    // Scatter the rows of the matrix to all processes
    MPI_Scatter(mat, 4, MPI_INT, out, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Process the row
    for (j = 0; j < 4; j++) {
        out[j] = out[j] + rank + j;
    }

    // Gather the processed rows back into the root process
    MPI_Gather(out, 4, MPI_INT, mat, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Transformed Matrix:\n");
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++)
                printf("%d ", mat[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}