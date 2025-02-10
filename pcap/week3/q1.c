#include "mpi.h"
#include <stdio.h>
int factorial (int a){
    int j=1;
    for(int i=1;i<=a;i++){
        j=j*i;
    }
    return j;
}
int main(int argc, char *argv[]){
    int rank,size, N, A[10], c, b[10],i,ans;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank==0){
        N = size;
        fprintf(stdout, "Enter %d values: \n", N);
        fflush(stdout);
        for (i = 0; i < N; i++)
            scanf("%d", &A[i]);
    }
    MPI_Scatter(A,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
    c=factorial(c);
    MPI_Gather(&c,1,MPI_INT,b,1,MPI_INT, 0, MPI_COMM_WORLD);
if (rank == 0)
        fprintf(stdout, "The Result gathered in the root \n");
    fflush(stdout);
    for (i = 0; i < N; i++)
         ans+=b[i];
    fprintf(stdout, "%d \t", ans);
    fflush(stdout);    
    MPI_Finalize();
    return 0;





}