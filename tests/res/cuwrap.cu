void * devmalloc(size_t size) {
    void * d;

    cudaMalloc(&d, size);

    return d;
}

void * memcpy2host(void * h, void * d, size_t size) {

	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

    return h;
}

void devfree(void * d) {

	cudaFree(d);
}
