#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    double a = 10.0, b = 5.0, result;
    char operator;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        operator = '+'; 
        result = a + b;
        printf("Process %d: %lf %c %lf = %lf\n", rank, a, operator, b, result);
    } else if (rank == 1) {
        operator = '-';
        result = a - b;
        printf("Process %d: %lf %c %lf = %lf\n", rank, a, operator, b, result);
    } else if (rank == 2) {
        operator = '*'; 
        result = a * b;
        printf("Process %d: %lf %c %lf = %lf\n", rank, a, operator, b, result);
    } else if (rank == 3) {
        operator = '/'; 
        result = a / b;
        printf("Process %d: %lf %c %lf = %lf\n", rank, a, operator, b, result);
    }

    MPI_Finalize();
    return 0;
}
