#include "mpi.h"
#include <stdio.h>

double averagee(int *arr, int m)
{
    double sum = 0;
    for (int i = 0; i < m; i++)
    {
        sum += arr[i];
    }

    return sum / m;
}
int main(int argc, char *argv[])
{
    int rank, size, N, A[100], c[10], i, m;
    double d, b[100], ans=0.0;
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

    MPI_Scatter(A, m, MPI_INT, &c, m, MPI_INT, 0, MPI_COMM_WORLD);
    d = averagee(c, m);
    MPI_Gather(&d, 1, MPI_DOUBLE, b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (i = 0; i < size; i++)
            ans += b[i];
    double anss=ans/(size);
    fprintf(stdout, "the average is %f", anss);
    fflush(stdout);
    }

    
    MPI_Finalize();
    return 0;
}
