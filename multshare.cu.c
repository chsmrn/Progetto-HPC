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

#define BLOCK_SIZE 32
#define SEED 777

void MatMul(int *m1, int *m2, int *res, int m, int n, int p);
void MatInit(int *m, int rows, int cols);
void MatMulStreams(int *A, int *B, int *C, int m, int n, int p, int nstreams);
__global__ void MatInitKernel(int *m, int rows, int cols);
__global__ void MatMulKernel(int *m1, int *m2, int *res, int m, int n, int p);
__global__ void gpu_glob_matmul(int *m1,int *m2, int *res, int m, int n, int p);
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k);
void print_matrix(int *matrix, int r, int c);

int main(int argc, char *argv[])
{

    int *A, *B, *C, *C_cpu;
    int m, n, p;

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);

    A = (int *)malloc(m * n * sizeof(int));

    B = (int *)malloc(n * p * sizeof(int));

    C = (int *)malloc(m * p * sizeof(int));

    C_cpu = (int *)malloc(m * p * sizeof(int));

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    MatInit(A, m, n);
    MatInit(B, n, p);
    MatMulStreams(A, B, C, m, n, p,2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Elapsed time -> %f\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cpu_matrix_mult(A, B, C_cpu, m, n, p);

    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            if (C[i * p + j] != C_cpu[i * p + j])
            {
                printf("%d diverso da %d\n", C[i * p + j], C_cpu[i * p + j]);
                all_ok = 0;
            }
        }
    }

    if (all_ok)
    {
        printf("all results are correct!!!");
    }
    else
    {
        printf("incorrect results\n");
    }
}

void print_matrix(int *matrix, int r, int c)
{

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%d ", matrix[i * c + j]);
        }
        printf("\n");
    }
}

void MatMul(int *m1, int *m2, int *res, int m, int n, int p)
{
    // Load A and B to device memory
    int *d_A;
    size_t size = m * n * sizeof(int);
    cudaError_t err = cudaMalloc(&d_A, size);
    if (err)
        printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_A, m1, size, cudaMemcpyHostToDevice);
    if (err)
        printf("Copy A to device: %s\n", cudaGetErrorString(err));

    int *d_B;
    size = p * n * sizeof(int);
    err = cudaMalloc(&d_B, size);
    if (err)
        printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_B, m2, size, cudaMemcpyHostToDevice);
    if (err)
        printf("Copy B to device: %s\n", cudaGetErrorString(err));

    // Allocate C in device memory
    int *d_C;
    size = p * m * sizeof(int);
    err = cudaMalloc(&d_C, size);
    if (err)
        printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_C, res, size, cudaMemcpyHostToDevice);
    if (err)
        printf("Copy C to device: %s\n", cudaGetErrorString(err));
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridX, dimGridY;

    //Calcolo delle dimensioni
    if (p % dimBlock.x == 0)
    {
        dimGridX = p / dimBlock.x;
    }
    else
    {
        dimGridX = p / dimBlock.x + 1;
    }

    if (m % dimBlock.y == 0)
    {
        dimGridY = m / dimBlock.y;
    }
    else
    {
        dimGridY = m / dimBlock.y + 1;
    }

    dim3 dimGrid(dimGridX, dimGridY);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p);
    err = cudaDeviceSynchronize();

    if (err)
        printf("Run kernel: %s\n", cudaGetErrorString(err));

    // Read C from device memory
    err = cudaMemcpy(res, d_C, size, cudaMemcpyDeviceToHost);

    if (err)
    {
        printf("Copy C off of device: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void MatMulStreams(int *A, int *B, int *C, int m, int n, int p, int nstreams)
{
    // Load A and B to device memory
    /*
    if(err)
    printf("CUDA malloc C: %s\n",cudaGetErrorString(err)); 
    err = cudaMemcpy(d_C, res, size, cudaMemcpyHostToDevice); 
    if(err)
    printf("Copy C to device: %s\n",cudaGetErrorString(err)); 
    */
    // Set kernel size
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        int dimGridX, dimGridY;

        //Calcolo delle dimensioni
        if (p % dimBlock.x == 0)
        {
            dimGridX = p / dimBlock.x;
        }
        else
        {
            dimGridX = p / dimBlock.x + 1;
        }

        if (m % dimBlock.y == 0)
        {
            dimGridY = m / dimBlock.y;
        }
        else
        {
            dimGridY = m / dimBlock.y + 1;
        }

        dim3 dimGrid(dimGridX, dimGridY);
        
    cudaStream_t *streams;
    streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*nstreams);
    for( int i=0; i<nstreams; i++)
        cudaStreamCreate(&streams[i]);
    
    int *d_A;
    size_t size = m * n * sizeof(int);
    cudaError_t err = cudaMallocHost(&d_A, size);
    if (err)
        printf("CUDA malloc A: %s\n", cudaGetErrorString(err));

    /*
  err = cudaMemcpy(d_A, m1, size, cudaMemcpyHostToDevice); 
  if(err)
    printf("Copy A to device: %s\n",cudaGetErrorString(err)); 
  */

    int *d_B;
    size = p * n * sizeof(int);
    err = cudaMallocHost(&d_B, size);
    if (err)
        printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
    /*
    err = cudaMemcpy(d_B, m2, size, cudaMemcpyHostToDevice); 
  if(err)
   printf("Copy B to device: %s\n",cudaGetErrorString(err)); 
   */
    // Allocate C in device memory
    int *d_C;
    size = p * m * sizeof(int);
    err = cudaMallocHost(&d_C, size);
    if (err)
        printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
    
    int a_streamsize = m*n/nstreams;
    int b_streamsize = n*p/nstreams;
    int c_streamsize = m*p/nstreams;
    

    for (int i = 0; i < nstreams; i++)
    {

        unsigned int size_a = i==nstreams-1 ? (m*n - a_streamsize*(nstreams-1)) : a_streamsize;
        unsigned int size_b = i==nstreams-1 ? (p*n - b_streamsize*(nstreams-1)) : b_streamsize; 
        unsigned int size_c = i==nstreams-1 ? (m*p - c_streamsize*(nstreams-1)) : c_streamsize;

        printf("Sono stream %d, size sono %d %d %d\n",i,size_a,size_b,size_c);


        cudaMemcpyAsync(d_A+i*a_streamsize,A+i*a_streamsize,size_a*sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        cudaMemcpyAsync(d_B+i*b_streamsize,B+i*b_streamsize,size_b*sizeof(int), cudaMemcpyHostToDevice, streams[i]);


        gpu_glob_matmul<<<dimGrid, dimBlock,0,streams[i]>>>(d_A, d_B, d_C, m, n, p);

        
        cudaMemcpyAsync(d_C+i*c_streamsize, C+i*c_streamsize, size_c*sizeof(int) , cudaMemcpyDeviceToHost, streams[i]);
    }
        
        err = cudaDeviceSynchronize();

        if (err)
            printf("Run kernel: %s\n", cudaGetErrorString(err));


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}




void MatInit(int *m, int rows, int cols)
{

    int *m_dev;
    size_t size = rows * cols * sizeof(int);
    cudaError_t err = cudaMalloc(&m_dev, size);

    if (err)
        printf("CUDA malloc Matrix Dev: %s\n", cudaGetErrorString(err));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int dimGridX, dimGridY;

    //Calcolo delle dimensioni
    if (cols % dimBlock.x == 0)
    {
        dimGridX = cols / dimBlock.x;
    }
    else
    {
        dimGridX = cols / dimBlock.x + 1;
    }

    if (rows % dimBlock.y == 0)
    {
        dimGridY = rows / dimBlock.y;
    }
    else
    {
        dimGridY = rows / dimBlock.y + 1;
    }

    dim3 dimGrid(dimGridX, dimGridY);

    MatInitKernel<<<dimGrid, dimBlock>>>(m_dev, rows, cols);

    err = cudaDeviceSynchronize();

    if (err)
        printf("Run kernel: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(m_dev, m, size, cudaMemcpyHostToDevice);

    if (err)
        printf("Copy initialized matrix to device: %s\n", cudaGetErrorString(err));

    cudaFree(m_dev);
}

__global__ void MatInitKernel(int *m, int rows, int cols)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows * cols)
    {
        return;
    }

    m[i] = (i * blockIdx.x + blockDim.x + SEED * threadIdx.x) % 10;
}

__global__ void MatMulKernel(int *m1, int *m2, int *res, int m, int n, int p)
{
    //Temporary variable to hold the thread-local result
    int CValue = 0;

    //indice di riga e di colonna di ogni thread, dove inizia il quadrato
    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    //Submatrices stored in shared memory to hold part of the input matrices
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    //Divide number of columns of m1 through block's threads
    for (int k = 0; k < (BLOCK_SIZE + n - 1) / BLOCK_SIZE; k++)
    {

        //Go to k-th block with stride of thread column and check if Row is inside of the input matrix m1
        if (k * BLOCK_SIZE + threadIdx.x < n && Row < m)
            As[threadIdx.y][threadIdx.x] = m1[Row * n + k * BLOCK_SIZE + threadIdx.x];
        else
            //If Row or threadIdx.x strided element of k-th block exceeded the dimenion of the first matrix, set element to 0 (padding)
            As[threadIdx.y][threadIdx.x] = 0;

        //Go to k-th block with stride of thread row and check if Col is inside of input matrix m2
        if (k * BLOCK_SIZE + threadIdx.y < n && Col < p)
            Bs[threadIdx.y][threadIdx.x] = m2[(k * BLOCK_SIZE + threadIdx.y) * p + Col];
        else
            //If Col or threadIdx.y strided element of k-th block exceeded the dimenion of the second matrix, set element to 0 (padding)
            Bs[threadIdx.y][threadIdx.x] = 0;

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
    {
        //printf("Srivo in d_C alla posizione %d l'elemento %d\n",((blockIdx.y * blockDim.y + threadIdx.y)*p) +
        //   (blockIdx.x * blockDim.x)+ threadIdx.x,CValue);
        res[((blockIdx.y * blockDim.y + threadIdx.y) * p) +
            (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
    }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k)
{
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