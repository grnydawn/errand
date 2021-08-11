#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int dim;

__global__ void test(int *gpu_Num){
    *gpu_Num = dim;
}

int main(int argc, char* argv[])
{
    int num = 25;
    cudaMemcpyToSymbol(dim,&num,sizeof(int),0,cudaMemcpyHostToDevice);
    
    int *gpu_Num;
    cudaMalloc(&gpu_Num,sizeof(int));

    test<<<1,1>>>(gpu_Num);

    int hostResult;
    cudaMemcpy(&hostResult,gpu_Num,sizeof(int),cudaMemcpyDefault);
    
    printf("Result: %i\n",hostResult);
}
