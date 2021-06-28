/*
 * multShare.c
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */
#include <stdio.h>
#include <stdlib.h>


__global__ void MatMulKernel(int* m1, int* m2, int* res, int m, int n,  int p);
__global__ void MatInitKernel(int *m, int rows, int cols);
__global__ void gpu_texture_matmul(int *m1,int *m2, int *res, int m, int n, int p);
void cpu_matrix_mult(int *m1,int *m2, int *res, int m, int n, int p);
void MatMul(int* m1, int* m2, int* res, int m, int n, int p);
void MatMulTex(int* m1, int* m2, int* res, int m, int n, int p);
void print_matrix(char *id, int* mat, int rows, int cols);

#define BLOCK_SIZE 32

texture<int,1> texRef_m1;
texture<int,1> texRef_m2;


int main(int argc, char* argv[]){
  
  int  *A, *B, *C, *C_cpu;
  
  int m,n,p;
  
  m = atoi(argv[1]);			
  n = atoi(argv[2]);			
  p = atoi(argv[3]);			

  A  = (int*)malloc(m * n * sizeof(int));

  B  = (int*)malloc(n * p * sizeof(int));

  C  = (int*)malloc(m * p * sizeof(int));

  C_cpu  = (int*)malloc(m * p * sizeof(int));


  for(int i = 0; i < m; i++)
    for(int j = 0; j < n; j++)
      A[i*n + j] = (rand() % 3);

  for(int i = 0; i < n; i++)
    for(int j = 0; j < p; j++)
      B[i*p + j] = (rand() % 2);

  MatMulTex(A, B, C, m, n, p);

  cpu_matrix_mult(A,B,C_cpu,m,n,p);


  int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if(C[i*p + j] != C_cpu[i*p + j])
            {
                all_ok = 0;
            }
        }
    }

    if(all_ok)
    {
        printf("all results are correct!!!");
    }
    else
    {
        printf("incorrect results\n");
    }

}


void print_matrix(char *id, int* mat, int rows, int cols){

  printf("%s\n",id);
  
  for( int i=0; i<rows; i++){
    
    for( int j=0; j<cols; j++){

      printf("%d ",mat[i*cols+j]);
    
    }
  
    printf("\n");
  
  }


}

void MatMulTex(int* A, int* B, int* C, int m, int n, int p){
    // Load A and B to device memory 
  int *d_A; 
  size_t size = m * n * sizeof(int); 
  cudaError_t err = cudaMalloc(&d_A, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 

  int *d_B; 
  size = p * n * sizeof(int); 
  err = cudaMalloc(&d_B, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); 
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  // Allocate C in device memory 
  int *d_C; 
  size = p * m * sizeof(int); 
  err = cudaMalloc(&d_C, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice); 
  printf("Copy C to device: %s\n",cudaGetErrorString(err)); 


  // Invoke kernel 
    
  cudaChannelFormatDesc desc_a = cudaCreateChannelDesc<int>();
  cudaBindTexture(0,texRef_m1,d_A,desc_a);

  cudaChannelFormatDesc desc_b = cudaCreateChannelDesc<int>();
  cudaBindTexture(0,texRef_m2,d_B,desc_b);
  



  unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (p + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
  printf("Assegno %d righe e %d colonne\n",grid_rows, grid_cols);
    
  dim3 dimGrid(grid_cols, grid_rows);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
  gpu_texture_matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p); 
    
  err = cudaDeviceSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory 
  err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 

  
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  //print_matrix("Risultato device", d_C, m, p);

  //printf("%d",d_C[0]);
  //printf("%d",C[0]);

  //print_matrix("Risultato host",C, m, p);

  // Free device memory
  cudaFree(d_A); 
  cudaFree(d_B); 
  cudaFree(d_C);

  cudaUnbindTexture(texRef_m1);
  cudaUnbindTexture(texRef_m2);
}


__global__ void MatInitKernel(int *m, int rows, int cols){
  return;
}

__global__ void MatMulKernel(int* A, int* B, int* C, int m, int n, int p)
{
    //Temporary variable to hold the thread-local result
    int CValue = 0;

    //indice di riga e di colonna di ogni thread, dove inizia il quadrato
    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    //Submatrices stored in shared memory to hold part of the input matrices
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];


    //Divide number of columns of m1 through block's threads
    for (int k = 0; k < (BLOCK_SIZE + n - 1)/BLOCK_SIZE; k++) {

        //Go to k-th block with stride of thread column and check if Row is inside of the input matrix m1
         if (k*BLOCK_SIZE + threadIdx.x < n && Row < m){
             
             As[threadIdx.y][threadIdx.x] = A[Row*n + k*BLOCK_SIZE + threadIdx.x];
         }
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
        C[((blockIdx.y * blockDim.y + threadIdx.y)*p) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}


void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
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

__global__ void gpu_texture_matmul(int *m1,int *m2, int *res, int m, int n, int p)
{ 
    //Mapping between row and column of the result matrix and GPU thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //Temporary variable to store the result of a single result matrix cell
    int temp_res = 0;
    //Since more threads that needed can be spawned, check if row and column values are in-bound
    if( col < p && row < m) 
    {
        //Perform row-column product providing the result of a single result cell scrolling through n columns of the selected row of the first
        //matrix and the n rows of the selected column of the second matrix
        for(int i = 0; i < n; i++) 
        {
            temp_res += tex1Dfetch(texRef_m1,row * n + i)*tex1Dfetch(texRef_m2,i * p + col);
            //temp_res += m1[row * n + i] * m2[i * p + col];
        }
        res[row * p + col] = temp_res;
    }
}