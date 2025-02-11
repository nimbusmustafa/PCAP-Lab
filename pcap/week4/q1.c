#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long factorial(int num) {
    long long fact = 1;
    for (int i = 1; i <= num; i++) {
        fact *= i;
    }
    return fact;
}

// Error handler function
void handle_error(int errcode) {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_len;
    MPI_Error_string(errcode, err_string, &err_len);
    fprintf(stderr, "MPI Error: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, errcode);
}

int main(int argc, char *argv[]) {
    int rank, size, N, err_code;
    long long local_fact, scan_sum;

    // Initialize MPI
    err_code = MPI_Init(&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    N=size;
    for(int j=1;j<=rank+1;j++)
    local_fact = factorial(j);


    MPI_Scan(&local_fact, &scan_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (rank == N - 1) {
        printf("Sum of factorials (1! + 2! + ... + %d!) = %lld\n", N, scan_sum);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
