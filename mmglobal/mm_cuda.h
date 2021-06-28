#include <stdio.h>
#include <stdlib.h>


__global__ void gpu_glob_matmul(int *m1,int *m2, int *res, int m, int n, int p);

