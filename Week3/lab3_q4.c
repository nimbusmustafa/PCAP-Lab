#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, n, i;
    char S1[100], S2[100], Result[200];  // Resultant string array of twice the size for combined chars

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads the input strings
        printf("Enter the first string (S1): ");
        scanf("%s", S1);
        printf("Enter the second string (S2): ");
        scanf("%s", S2);

        // Check if both strings are of the same length
        int len1 = strlen(S1);
        int len2 = strlen(S2);

        if (len1 != len2) {
            printf("Error: Strings must be of the same length!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = len1 / size; // Number of characters per process
    }

    // Broadcast the number of characters per process (n)
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the corresponding portions of both strings S1 and S2 to all processes
    char local_S1[n], local_S2[n];
    MPI_Scatter(S1, n, MPI_CHAR, local_S1, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, n, MPI_CHAR, local_S2, n, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Each process combines the corresponding characters from local_S1 and local_S2
    char local_Result[2 * n + 1];  // local result string (twice the length of the portion)
    for (i = 0; i < n; i++) {
        local_Result[2 * i] = local_S1[i];      // Add character from S1
        local_Result[2 * i + 1] = local_S2[i];  // Add character from S2
    }
    local_Result[2 * n] = '\0';  // Null-terminate the local result string

    // Gather the partial results from all processes into the root
    MPI_Gather(local_Result, 2 * n, MPI_CHAR, Result, 2 * n, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process displays the final resultant string
    if (rank == 0) {
        printf("\nResultant String: %s\n", Result);
    }

    MPI_Finalize();
    return 0;
}
