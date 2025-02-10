#include <stdio.h>
#include <ctype.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char str[] = "HELLO";
    char toggledChar;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank < 5) {
        toggledChar = (isupper(str[rank])) ? tolower(str[rank]) : toupper(str[rank]);
        printf("Process %d toggled character: %c -> %c\n", rank, str[rank], toggledChar);
    }

    MPI_Finalize();
    return 0;
}