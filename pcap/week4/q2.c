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

// Error handler function
void handle_error(int errcode, const char *msg) {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_len;

    MPI_Error_string(errcode, err_string, &err_len);
    fprintf(stderr, "MPI Error in %s: %s\n", msg, err_string);
    MPI_Abort(MPI_COMM_WORLD, errcode);
}

int main(int argc, char *argv[]) {
    int rank, size, err_code;
    int matrix[3][3]; // 3x3 matrix
    int target, local_count = 0, global_count = 0;
    int rows_per_process = 1;  // 3 rows, 3 processes, each process gets 1 row
    
    // Initialize MPI
    err_code = MPI_Init(&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Init");
    }

    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Comm_rank");
    }

    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Comm_size");
    }

    // Ensure we have exactly 3 processes
    if (size != 3) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 3 processes.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Root process (rank 0) reads the matrix and target
    if (rank == 0) {
        printf("Enter the elements of a 3x3 matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("matrix[%d][%d]: ", i, j);
                if (scanf("%d", &matrix[i][j]) != 1) {
                    fprintf(stderr, "Error: Invalid input for matrix[%d][%d]\n", i, j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }

        // Ask for the element to search
        printf("Enter the element to search: ");
        if (scanf("%d", &target) != 1) {
            fprintf(stderr, "Error: Invalid input for target element.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the target element to all processes
    err_code = MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Bcast");
    }

    // Scatter the rows of the matrix to all processes
    int local_row[3]; // Local row for each process
    err_code = MPI_Scatter(matrix, 3, MPI_INT, local_row, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Scatter");
    }

    // Each process counts the occurrences of the target in its row
    local_count = count_occurrences(local_row, target, 3);

    // Perform an MPI_Reduce to sum the local counts from all processes
    err_code = MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Reduce");
    }

    // Root process prints the result
    if (rank == 0) {
        printf("The element %d appears %d times in the matrix.\n", target, global_count);
    }

    // Finalize MPI
    err_code = MPI_Finalize();
    if (err_code != MPI_SUCCESS) {
        handle_error(err_code, "MPI_Finalize");
    }

    return 0;
}
