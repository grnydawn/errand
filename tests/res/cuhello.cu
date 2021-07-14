#include "stdio.h"

extern void * devmalloc(size_t size);
extern void * memcpy2host(void * h, void * d, size_t size);
extern void devfree(void * d);

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
	int a,b,c;
	int *dev_c;

	a=3;
	b=4;

	dev_c = (int *) devmalloc(sizeof(int));

	add<<<1,1>>>(a,b,dev_c);

    memcpy2host(&c, dev_c, sizeof(int));

	printf("%d + %d is %d\n", a, b, c);

	devfree(dev_c);

	return 0;
}

