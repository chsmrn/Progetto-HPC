#include "mm_cuda.h"
#include <stdlib.h>


/**
 * This function provides the kernel to perform the matrix-matrix product using Global memory approach in CUDA
 * Input parameters are: m1,m2 (input matrices), res (output matrices), m,n,p (sizes of the problem since the product is performed
 * between a (m,n) and a (n,p) matrices ).
 * The function returns void as every CUDA kernel function.
 */

__global__ void gpu_glob_matmul(int *m1,int *m2, int *res, int m, int n, int p)
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
            temp_res += m1[row * n + i] * m2[i * p + col];
        }
        res[row * p + col] = temp_res;
    }
} 


texture<int, 1> texRef_m1; 
texture<int, 1> texRef_m2; 


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
            temp_res += tex1dfetch(texRef_m1,row * n + i)*tex1dfetch(texRef_m2,i * p + col);
            //temp_res += m1[row * n + i] * m2[i * p + col];
        }
        res[row * p + col] = temp_res;
    }
} 

void MatMulTex(int *m1, int *m2, int *res, int m, int n, int p){

    cudaEvent_t start, stop;
   int *d_m1;
   size_t size = m * n * sizeof(int); 
   cudaMalloc((void**)&d_m1, size); 
   cudaMemcpy(d_m1, m1, size, cudaMemcpyHostToDevice); 
   
   int *d_m2;
   size_t size = p * n * sizeof(int); 
   cudaMalloc((void**)&d_m2, size); 
   cudaMemcpy(d_m2, m2, size, cudaMemcpyHostToDevice); 
   

   // Allocate C in device memory 
   int *d_res;
   size_t size = m * p * sizeof(int); 
   cudaMalloc((void**)&d_res, size); 
   
   //Texture channel descriptor
//   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
   // Bind A_tex to A
   cudaError_t  errt = cudaBindTexture(0,texRef_m1,d_m1,channelDesc);
   if( errt != cudaSuccess) printf("can not bind to texture \n");
   // Bind B_tex to B
   errt = cudaBindTexture(0,texRef_m2,d_m2,channelDesc);
   if( errt != cudaSuccess) printf("can not bind to texture \n");
   
    // Grid specify
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // Start timing
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start,0);

// Invoke kernel 
   gpu_texture_matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,m,n,p);

// End timing
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float transferTime;
   cudaEventElapsedTime(&transferTime, start, stop);
   double dSeconds = transferTime/((double)1000.0);
   double dNumOps = 2.0 * (double)n * (double)m * (double)p;
   double gflops = 1.0e-9 * dNumOps/dSeconds;
   printf("CUDA Gflops = %.4f , Time = %.5f s dim=%d\n", gflops, dSeconds,n);

   // Read C from device memory 
   cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost); 

   // Unbind texture
   cudaUnbindTexture ( texRef_m1 ) ;
   cudaUnbindTexture ( texRef_m2 ) ;

   // Free device memory 
   cudaFree(d_m1);    
   cudaFree(d_m2); 
   cudaFree(d_res); 
   cudaThreadExit();



}




