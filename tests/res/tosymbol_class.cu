#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


class A {
public:
	int x;
};

A * var;

__global__ void test(A * dvar){
	dvar->x = 1;
}

int main(int argc, char* argv[])
{
    //int num = 25;
    //cudaMemcpyToSymbol(dim,&num,sizeof(int),0,cudaMemcpyHostToDevice);
    
    //int *gpu_Num;
    cudaMalloc(&var,sizeof(A));

    test<<<1,1>>>(var);

	int p;
    cudaMemcpy(&p,&(var->x),sizeof(int),cudaMemcpyDefault);
    
    printf("Result: %i\n",p);
}
