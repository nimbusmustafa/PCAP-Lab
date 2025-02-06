#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void err_handle(int err_code, int rank)
{
    if (err_code != MPI_SUCCESS)
    {
        char err_string[1000];
        int length, err_class;
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_code, err_string, &length);
        if(rank < 0){
            fprintf(stderr, "MPI_Init method failed\n");
        }
        else{
            fprintf(stderr, "MPI error in rank %d\n", rank);
        }
        fprintf(stderr, "MPI Error string %s\n", err_string);
        fprintf(stderr, "MPI error class %d\n", err_class);
        fprintf(stderr, "MPI error code %d\n", err_code);
    }
    return;
}

int main(int argc, char **argv)
{
    int rank, size, *revcount, *displace;
    char *input, *ans, *store, ele;
    int err_code = MPI_Init(&argc, &argv);
    err_handle(err_code, -1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    // note when using bcast the space should be allocated for all process if sending an array. Else error will come
    input = (char *)malloc((size + 1) * sizeof(char));
    revcount = (int *)malloc(size * sizeof(int));
    displace = (int *)malloc(size * sizeof(int));
    if (rank == 0)
    {
        printf("enter string\n");
        scanf("%s", input);
        int count = 0;
        for (int i = 0; i < size; ++i)
        {
            revcount[i] = i + 1;
            displace[i] = count;
            count += revcount[i];
        }
        ans = (char *)malloc((count + 1) * sizeof(char));
        ans[count] = '\0';
    }
    err_code = MPI_Bcast(input, size + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    err_handle(err_code, -1);
    err_code = MPI_Bcast(revcount, size, MPI_INT, 0, MPI_COMM_WORLD);
    err_handle(err_code, -1);
    err_code = MPI_Bcast(displace, size, MPI_INT, 0, MPI_COMM_WORLD);
    err_handle(err_code, -1);
    store = (char *)malloc((rank + 1) * sizeof(char));
    ele = input[rank];
    for (int i = 0; i < rank + 1; ++i)
    {
        store[i] = ele;
    }
    err_code = MPI_Gatherv(store, rank + 1, MPI_CHAR, ans, revcount, displace, MPI_CHAR, 0, MPI_COMM_WORLD);
    err_handle(err_code, -1);
    if (rank == 0)
    {
        printf("%s is the answer\n", ans);
    }
    MPI_Finalize();
    return 0;
}