#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, n, i;
    char S1[100], S2[100], Result[200];  // Resultant string array of twice the size

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the first string (S1): ");
        scanf("%s", S1);
        printf("Enter the second string (S2): ");
        scanf("%s", S2);

        int len1 = strlen(S1);
        int len2 = strlen(S2);

        if (len1 != len2) {
            printf("Error: Strings must be of the same length!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = len1 / size;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char local_S1[n], local_S2[n];
    MPI_Scatter(S1, n, MPI_CHAR, local_S1, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, n, MPI_CHAR, local_S2, n, MPI_CHAR, 0, MPI_COMM_WORLD);

    char local_Result[2 * n];  
    for (i = 0; i < n; i++) {
        local_Result[2 * i] = local_S1[i];      
        local_Result[2 * i + 1] = local_S2[i];  
    }

    MPI_Gather(local_Result, 2 * n, MPI_CHAR, Result, 2 * n, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Result[2 * n * size] = '\0';  // ✅ Explicitly null-terminate the final string
        printf("\nResultant String: %s\n", Result);
    }

    MPI_Finalize();
    return 0;
}
