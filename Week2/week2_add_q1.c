#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int is_prime(int num) {
    if (num <= 1) return 0;  
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) {
            return 0; 
        }
    }
    return 1;  
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    int *array = NULL;
    int value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least two processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        // Master process: Read the number of elements (N) and the array values
        printf("Enter the number of elements: ");
        scanf("%d", &N);
        
        array = (int *)malloc(N * sizeof(int));
        printf("Enter %d elements:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &array[i]);
        }

        // Scatter the array elements to all processes
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(array, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // Non-root processes
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Check if the received value is prime
    int result = is_prime(value);

    // Gather the results from all processes
    int *results = NULL;
    if (rank == 0) {
        results = (int *)malloc(N * sizeof(int));
    }
    MPI_Gather(&result, 1, MPI_INT, results, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the results
    if (rank == 0) {
        printf("\nPrime Check Results:\n");
        for (int i = 0; i < N; i++) {
            if (results[i] == 1) {
                printf("Element %d: %d is Prime\n", i, array[i]);
            } else {
                printf("Element %d: %d is Not Prime\n", i, array[i]);
            }
        }
        free(results);  
        free(array);    
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
