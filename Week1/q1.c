#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int x=3;
   // scanf("%d",&x);
    double result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    result = pow(x, rank);
    printf("Process %d: pow(%d, %d) = %f\n", rank, x, rank, result);

    MPI_Finalize();
    return 0;
}
