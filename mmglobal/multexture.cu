#include <stdio.h>
#include <assert.h>

#define epsilon (float)1e-5
#define verbose 0
#define check 1

// Thread block size
#define NB 16

texture<float,1>  A_tex;
texture<float,1>  B_tex;




// Get a texture matrix element 
//__device__ float GetTexElement(const texture<float, 1> texref, const Matrix A, int Asub, int row, int col)
//{
//   return tex1Dfetch(texref,Asub + row * A.stride + col);
//}

// Get a texture matrix element 
__device__ float GetTexElement_A(const Matrix A, int Asub, int row, int col)
{
  return tex1Dfetch(A_tex,Asub + row * A.stride + col);
}

// Get a texture matrix element 
__device__ float GetTexElement_B(const Matrix A, int Asub, int row, int col)
{
   return tex1Dfetch(B_tex,Asub + row * A.stride + col);
}


// Set a matrix element 
__device__ void SetElement(Matrix A, int row, int col, float value) 
{ 
   A.elements[row * A.stride + col] = value; 
}

// Get the NBxNB sub-matrix Asub of A that is 
// located col sub-matrices to the right and row sub-matrices down 
// from the upper-left corner of A 
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
   Matrix Asub; 
   Asub.width = NB; 
   Asub.height = NB; 
   Asub.stride = A.stride; 
   Asub.elements = &A.elements[A.stride * NB * row + NB * col];
   return Asub; 
}

// Forward declaration of the matrix multiplication kernel 
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix); 

// Matrix multiplication - Host code 
// Matrix dimensions are assumed to be multiples of NB 
void MatMul(const Matrix A, const Matrix B, Matrix C) 
{ 
   cudaEvent_t start, stop;
   // Load A and B to device memory 
   float *A;
   d_A.width = d_A.stride = A.width; d_A.height = A.height; 
   size_t size = A.width * A.height * sizeof(float); 
   cudaMalloc((void**)&d_A.elements, size); 
   cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
   Matrix d_B; 
   d_B.width = d_B.stride = B.width; d_B.height = B.height; 
   size = B.width * B.height * sizeof(float); 
   cudaMalloc((void**)&d_B.elements, size); 
   cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

   // Allocate C in device memory 
   Matrix d_C; 
   d_C.width = d_C.stride = C.width; d_C.height = C.height; 
   size = C.width * C.height * sizeof(float); 
   cudaMalloc((void**)&d_C.elements, size);

   //Texture channel descriptor
//   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   // Bind A_tex to A
   cudaError_t  errt = cudaBindTexture(0,A_tex,d_A.elements,channelDesc);
   if( errt != cudaSuccess) printf("can not bind to texture \n");
   // Bind B_tex to B
   errt = cudaBindTexture(0,B_tex,d_B.elements,channelDesc);
   if( errt != cudaSuccess) printf("can not bind to texture \n");
   
// Grid specify
   dim3 dimBlock(NB, NB); 
   dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.x);
   MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

// Start timing
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start,0);

// Invoke kernel 
   MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

// End timing
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float transferTime;
   cudaEventElapsedTime(&transferTime, start, stop);
   double dSeconds = transferTime/((double)1000.0);
   double dNumOps = 2.0 * (double)A.width * (double)A.height * (double)B.width;
   double gflops = 1.0e-9 * dNumOps/dSeconds;
   printf("CUDA Gflops = %.4f , Time = %.5f s dim=%d\n", gflops, dSeconds,A.width);

   // Read C from device memory 
   cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 

   // Unbind texture
   cudaUnbindTexture ( A_tex ) ;
   cudaUnbindTexture ( B_tex ) ;

   // Free device memory 
   cudaFree(d_A.elements); 
   cudaFree(d_B.elements); 
   cudaFree(d_C.elements);
   cudaThreadExit();
}

// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernel(const Matrix A,const Matrix B, Matrix C) 
{
   // Shared memory used to store Asub and Bsub respectively
   __shared__ float As[NB][NB];
   __shared__ float Bs[NB][NB];
 
   // Block row and column 
   int ib = blockIdx.y; 
   int jb = blockIdx.x; 

   // Thread row and column within Csub 
   int it = threadIdx.y; 
   int jt = threadIdx.x; 

   // Each thread computes one element of Csub 
   // by accumulating results into Cvalue 
   float Cvalue = 0; 

   // Loop over all the sub-matrices of A and B that are 
   // required to compute Csub 
   // Multiply each pair of sub-matrices together 
   // and accumulate the results 

    As[it][jt] = GetTexElement_A(A,Asub, it, jt); 
    Bs[it][jt] = GetTexElement_B(B,Bsub, it, jt); 

   for (int kb = 0; kb < (A.width / NB); ++kb) {
      // Get sub-matrix Asub of A 
      int  Asub = A.stride * NB * ib + NB * kb;
      // Get sub-matrix Bsub of B 
      int Bsub = A.stride * NB * kb + NB * jb;
    
      // Load Asub and Bsub from device texture memory to shared memory 
      // Each thread loads one element of each sub-matrix 
      
      
      // Synchronize to make sure the sub-matrices are loaded 
      // before starting the computation 
      __syncthreads();

      // Multiply As and Bs together 
      for (int k = 0; k < NB; ++k) {
         Cvalue += As[it][k] * Bs[k][jt]; 
      }
      // Synchronize to make sure that the preceding 
      // computation is done before loading two new 
      // sub-matrices of A and B in the next iteration 
      __syncthreads(); 
   } 
   
   // Each thread block computes one sub-matrix Csub of C 
   Matrix Csub = GetSubMatrix(C, ib, jb);
   // Write Csub to device memory 
   // Each thread writes one element 
   SetElement(Csub, it, jt, Cvalue); 
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


// main

int main(int argc, char** argv) {

  int dim = 64*NB;
  Matrix h_A;
  h_A.width = dim; h_A.height = dim;
  size_t size = h_A.width*h_A.height*sizeof(float);
  h_A.elements = (float*)malloc(size);

  Matrix h_B;
  h_B.width = dim; h_B.height = dim;
  size = h_B.width*h_B.height*sizeof(float);
  h_B.elements = (float*)malloc(size);


  Matrix h_C;
  h_C.width = h_A.height; h_C.height = h_B.width;
  size = h_C.width*h_C.height*sizeof(float);
  h_C.elements = (float*)malloc(size);

  Matrix gold_C;
  gold_C.width = h_A.height; gold_C.height = h_B.width;
  gold_C.elements = (float*)malloc(size);

  randomInit(h_A.elements, h_A.width*h_A.height);

  if(verbose){
  // print h_A
  printf("\nh_A\n");
  for(int y=0; y<h_A.height; y++){
   printf("\n");
   for(int x=0; x<h_A.width; x++) {
      printf("%f ", h_A.elements[y*h_A.width+x]);
   }
  }
  }

  randomInit(h_B.elements, h_B.width*h_B.height);

  if(verbose){
   // print h_B
   printf("\nh_B\n");
   for(int y=0; y<h_B.height; y++){
    printf("\n");
    for(int x=0; x<h_B.width; x++) {
       printf("%f ", h_B.elements[y*h_B.width+x]);
    }
   }
  }

  for(int y=0; y<h_C.height; y++)
   for(int x=0; x<h_C.width; x++) {
      h_C.elements[y*h_C.width+x] = 0.0;
   }


  MatMul(h_A,h_B,h_C);
 if(check){
  for(int y=0; y<h_A.height; y++)
   for(int x=0; x<h_B.width; x++) {
    float Cvalue = (float)0.0;
    for(int k=0; k<h_B.height; k++)
      Cvalue +=  h_A.elements[y*h_A.width+k] * h_B.elements[k*h_B.width+x];
    gold_C.elements[y*gold_C.width+x] = Cvalue;
  }
 }
  if(verbose){
   // print gold_C
   printf("\n\ngold_C");
   for(int y=0; y<gold_C.height; y++){
    printf("\n");
    for(int x=0; x<gold_C.width; x++) {
       printf("%f ", gold_C.elements[y*gold_C.width+x]);
    }
   }

   // print h_C
   printf("\n\n\nh_C");
   for(int y=0; y<h_C.height; y++){
    printf("\n");
    for(int x=0; x<h_C.width; x++) {
       printf("%f ", h_C.elements[y*h_C.width+x]);
    }
   }
  }

 if(check){
  int errCnt = 0;
  for(int y=0; y<gold_C.height; y++){
   for(int x=0; x<gold_C.width; x++) {
      float it = gold_C.elements[y*gold_C.width+x];
      if(fabs(it - h_C.elements[y*h_C.width+x])> epsilon*it)
       errCnt++;

   }
  }

  if(errCnt==0)
    printf("\nTEST PASSED\n");
  else
    printf("\n\nTEST FAILED: number of errors:  %d\n", errCnt);
 }
  free(h_A.elements);
  free(h_B.elements);
  free(h_C.elements);
  free(gold_C.elements);

}