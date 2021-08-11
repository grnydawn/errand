#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class A {
public:
	int * x;
};

A var = A();

__global__ void test(A & dvar){

    //dvar.x = y;

	dvar.x[0] = 10;
	dvar.x[1] = 20;
}

int main(int argc, char* argv[])
{
    cudaMalloc(&(var.x),sizeof(int)*2);

    test<<<1,1>>>(var);

	int p[2];
    cudaMemcpy(p,var.x,sizeof(int)*2,cudaMemcpyDefault);
    
    printf("Result: %i, %i\n",p[0], p[1]);
}
