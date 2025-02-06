#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Error handling function
void error_handle(int err_code, int rank, const char *func_name)
{
    if (err_code != MPI_SUCCESS)
    {
        char err_string[1000];
        int length, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, err_string, &length);
        
        if (rank >= 0)
        {
            fprintf(stderr, "MPI Error for rank %d in function %s\n", rank, func_name);
        }
        else
        {
            fprintf(stderr, "MPI Initialization failed in function %s\n", func_name);
        }
        
        fprintf(stderr, "MPI Error string: %s\n", err_string);
        fprintf(stderr, "MPI Error class: %d\n", err_class);
        fprintf(stderr, "MPI Error code: %d\n", err_code);
    }
}

// Function to compute factorial of a number
long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char **argv)
{
    int rank, size, N;
    long long fact = 0, ans = 0;
    int err_code;

    // Initialize MPI
    err_code = MPI_Init(&argc, &argv);
    error_handle(err_code, -1, "MPI_Init");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set error handler to return instead of abort
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // Get the value of N from the root process
    if (rank == 0)
    {
        printf("Enter the value of N (number of factorials to sum): ");
        fflush(stdout);
        if (scanf("%d", &N) != 1 || N <= 0)
        {
            fprintf(stderr, "Error: Invalid input for N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast N to all processes
    err_code = MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    error_handle(err_code, rank, "MPI_Bcast");

    // Ensure that N is less than or equal to the number of processes
    if (N > size)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Error: N is greater than the number of processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Each process computes its portion of the factorials
    if (rank < N) // Only the first N processes should do work
    {
        // Calculate factorial for its corresponding rank (adjusted by 1)
        fact = factorial(rank + 1); // Calculate rank+1 factorial
        
        // Print the local factorial (optional, can be disabled for large N)
        printf("Rank %d: %d! = %lld\n", rank, rank + 1, fact);
    }

    // Perform the reduction to sum the factorials from rank 0 to N-1
    err_code = MPI_Reduce(&fact, &ans, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    error_handle(err_code, rank, "MPI_Reduce");

    // Only the root process (rank 0) will print the final result
    if (rank == 0)
    {
        printf("The sum of factorials from 1! to %d! is: %lld\n", N, ans);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
