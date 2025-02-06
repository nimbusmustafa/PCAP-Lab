#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to count occurrences of a target in a given row
int count_occurrences(int *row, int target, int cols) {
    int count = 0;
    for (int i = 0; i < cols; i++) {
        if (row[i] == target) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[3][3]; // 3x3 matrix
    int target, local_count = 0, global_count = 0;
    int rows_per_process = 1;  // 3 rows, 3 processes, each process gets 1 row
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process (rank 0) reads the matrix and target
    if (rank == 0) {
        // Read matrix from the user
        printf("Enter the elements of a 3x3 matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("matrix[%d][%d]: ", i, j);
                scanf("%d", &matrix[i][j]);
            }
        }

        // Ask for the element to search
        printf("Enter the element to search: ");
        scanf("%d", &target);
    }

    // Broadcast the target element to all processes
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the rows of the matrix to all processes
    int local_row[3]; // Local row for each process
    MPI_Scatter(matrix, 3, MPI_INT, local_row, 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts the occurrences of the target in its row
    local_count = count_occurrences(local_row, target, 3);

    // Perform an MPI_Reduce to sum the local counts from all processes
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("The element %d appears %d times in the matrix.\n", target, global_count);
    }

    MPI_Finalize();
    return 0;
}
