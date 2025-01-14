#include <stdio.h>
#include <mpi.h>

int factorial(int n) {
    if (n == 0) return 1;
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int fibonacci(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    int a = 0, b = 1, temp;
    for (int i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank % 2 == 0) {
        printf("Process %d: Factorial of %d is %d\n", rank, rank, factorial(rank));
    } else {
        printf("Process %d: Fibonacci of %d is %d\n", rank, rank, fibonacci(rank));
    }

    MPI_Finalize();
    return 0;
}
