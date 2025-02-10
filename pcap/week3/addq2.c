#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size, N, l, arr[100],arr2[100], n;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        fprintf(stdout, "enter length of arrray");
        fflush(stdout);
        scanf("%d", &l);
        fprintf(stdout, "enter elements");
        fflush(stdout);
        for (int i = 0; i < l; i++)
        {
            scanf("%d", &arr[i]);
        }

        n = l / size;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int local[n];
    MPI_Scatter(arr,n,MPI_INT,local,n,MPI_INT,0,MPI_COMM_WORLD  );
    for(int i=0;i<n;i++){
        if(local[i]%2==0){
            local[i]=1;
        }
        else local[i]=0;
    }
    MPI_Gather(local,n,MPI_INT,arr2,n,MPI_INT, 0, MPI_COMM_WORLD);
    for(int i=0;i<l;i++){
        fprintf(stdout , "%d\t", arr2[i]);
    }
    MPI_Finalize();
    return 0;
}