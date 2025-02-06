#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int rank, size, *arr, *store, *scan_result;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set error handler to return on error instead of aborting
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    // Root process (rank 0) handles matrix input
    if (rank == 0)
    {
        arr = (int *)malloc(16 * sizeof(int));
        printf("Enter the elements of the 4x4 matrix:\n");
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                scanf("%d", &arr[(4 * i) + j]);
            }
        }
    }

    // Allocate memory for the store and scan_result arrays
    store = (int *)malloc(4 * sizeof(int));
    scan_result = (int *)malloc(4 * sizeof(int));
    
    // Scatter the 4x4 matrix to all processes
    MPI_Scatter(arr, 4, MPI_INT, store, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform scan operation (prefix sum) for each element in store
    for (int i = 0; i < 4; ++i)
    {
        // Use scan_result to store the result to avoid aliasing
        MPI_Scan(&store[i], &scan_result[i], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    // Gather the results back to root process (rank 0)
    MPI_Gather(scan_result, 4, MPI_INT, arr, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the resulting matrix
    if (rank == 0)
    {
        printf("\nResulting matrix after scan operation:\n");
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                printf("%d ", arr[(4 * i) + j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
