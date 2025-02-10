#include "mpi.h"
#include <stdio.h>
#include <math.h>

void square(int *arr, int m, int power)
{
    for (int i = 0; i < m; i++)
    {
        arr[i] = pow(arr[i], power);
    }
}

int main(int argc, char *argv[])
{
    int rank, size, A[100], c[100], i, m;
    int b[100], ans = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        fprintf(stdout, "Enter m: \n");
        fflush(stdout);
        scanf("%d", &m);
        fprintf(stdout, "Enter %d values: \n", m * size);

        for (i = 0; i < m * size; i++)
            scanf("%d", &A[i]);
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, m, MPI_INT, c, m, MPI_INT, 0, MPI_COMM_WORLD);

    square(c, m, rank + 2);  

    MPI_Gather(c, m, MPI_INT, b, m, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (i = 0; i < m * size; i++)
        fprintf(stdout, "%d\n", b[i]);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
