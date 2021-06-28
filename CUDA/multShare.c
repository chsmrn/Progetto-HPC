/*
 * multShare.c
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */


#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void MatMulKernel(float* m1, float* m2, float* res, int m int n, int int p);
__global__ void MatInitKernel(float *m, int rows, int cols);
void MatMul(float* m1, float* m2, float* res, int m, int n, int p);




int main(int argc, char* argv[]){

  float  *A, *B, *C, *C_cpu;
  int m,n,p;
  m = atoi(argv[1]);			
  n = atoi(argv[2]);			
  p = atoi(argv[3]);			

  A  = (float*)malloc(m * n * sizeof(float));

  B  = (float*)malloc(n * p * sizeof(float));

  C  = (float*)malloc(m * p * sizeof(float));

  C_cpu  = (float*)malloc(m * p * sizeof(float));


  for(int i = 0; i < m; i++)
    for(int j = 0; j < n; j++)
      A[i*n + j] = (rand() % 3);

  for(int i = 0; i < n; i++)
    for(int j = 0; j < p; j++)
      B[i*p + j] = (rand() % 2);

  MatMul(A, B, C, m, n, p);

  cpu_matrix_mult(A,B,C_cpu,m,n,p);


  int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if(C[i*k + j] != C_cpu[i*k + j])
            {
                all_ok = 0;
            }
        }
    }

    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

}


void MatMul(float* m1, float* m2, float* res, int m, int n, int p) { 
  
  float *d_A; 
  size_t size = m * n * sizeof(float); 
  cudaError_t err = cudaMalloc(&d_A, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 

  float *d_B; 
  size = p * n * sizeof(float); 
  err = cudaMalloc(&d_B, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); 
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  float *d_C; 
  size = p * m * sizeof(float); 
  err = cudaMalloc(&d_B, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice); 
  printf("Copy C to device: %s\n",cudaGetErrorString(err)); 


  // Invoke kernel 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid( (p/dimBlock.x) , (m/dimBlock.y)); 
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p); 
  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory 
  err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory
  cudaFree(d_A); 
  cudaFree(d_B); 
  cudaFree(d_C); 
} 


__global__ void MatInitKernel(float *m, int rows, int cols){


  return;
}

__global__ void MatMulKernel(float* m1, float* m2, float* res, int m int n, int int p)
{
    //Temporary variable to hold the thread-local result
    float CValue = 0;

    //indice di riga e di colonna di ogni thread, dove inizia il quadrato
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    //Submatrices stored in shared memory to hold part of the input matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


    //Divide number of columns of m1 through block's threads
    for (int k = 0; k < (BLOCK_SIZE + n - 1)/BLOCK_SIZE; k++) {

        //Go to k-th block with stride of thread column and check if Row is inside of the input matrix m1
         if (k*BLOCK_SIZE + threadIdx.x < n && Row < m)
             As[threadIdx.y][threadIdx.x] = A[Row*n + k*BLOCK_SIZE + threadIdx.x];
         else
            //If Row or threadIdx.x strided element of k-th block exceeded the dimenion of the first matrix, set element to 0 (padding)
             As[threadIdx.y][threadIdx.x] = 0.0;

        //Go to k-th block with stride of thread row and check if Col is inside of input matrix m2
         if (k*BLOCK_SIZE + threadIdx.y < n && Col < p)
             Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*p + Col];
         else
            //If Col or threadIdx.y strided element of k-th block exceeded the dimenion of the second matrix, set element to 0 (padding)
             Bs[threadIdx.y][threadIdx.x] = 0.0;

        //Sync all threads to be sure that A and B submatrices are populated by all the threads
         __syncthreads();

        //Loop over the row and column assigned to the single thread to compute the local result and store it in Cvalue 
         for (int j = 0; j < BLOCK_SIZE; ++j)
             CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];

        //Second sync operationt to be sure that all the threads in the block computed the local result
         __syncthreads();
    }

    //Store the local result of each thread in the mapped position in the result matrix 
    if (Row < m && Col < p)
        res[((blockIdx.y * blockDim.y + threadIdx.y)*p) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}


void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}