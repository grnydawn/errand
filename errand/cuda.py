"""Errand CUDA engine module


"""

import os, sys, abc
import subprocess as subp

from numpy import double
from numpy.ctypeslib import load_library, ndpointer
from ctypes import c_double, c_size_t

from errand.engine import Engine
from errand.util import which


code_template = """
#include <stdio.h>
#include <unistd.h>

int isfinished = 0;

using namespace std;

// TODO: prepare all possible type/dim combinations
// dim: 0, 1,2,3,4,5
// type: int, float, char, boolean

{dvardefs}

{dvarcopyins}

{dvarcopyouts}

__global__ void kernel(double * a, double * b, double *c){{
    {devcodebody}
}}

extern "C" void h2dcopy_a(void * data, int size) {{
    h_a = (double *) data;
    h_a_size = size;
    cudaMalloc((void **)&(d_a), size * sizeof(double));
    cudaMalloc((void **)&d_a_size, sizeof(int));
    cudaMemcpy(d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_size, &(h_a_size), sizeof(int), cudaMemcpyHostToDevice);
    printf("BBBBBB %p, %f", d_a, h_a[0]);
}}

extern "C" void h2dcopy_b(void * data, int size) {{
    h_b = (double *) data;
    h_b_size = size;
    cudaMalloc((void **)&(d_b), size * sizeof(double));
    cudaMalloc((void **)&d_b_size, sizeof(int));
    cudaMemcpy(d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_size, &(h_b_size), sizeof(int), cudaMemcpyHostToDevice);
}}

extern "C" void h2dcopy_c(void * data, int size) {{
    h_c = (double *) data;
    h_c_size = size;
    cudaMalloc((void **)&(d_c), size * sizeof(double));
    cudaMalloc((void **)&d_c_size, sizeof(int));
    cudaMemcpy(d_c_size, &(h_c_size), sizeof(int), cudaMemcpyHostToDevice);
}}

extern "C" void d2hcopy_c(void * data, int size) {{
    cudaMemcpy(h_c, d_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    data = (void *) h_c;
}}

extern "C" int isalive() {{

    return isfinished;
}}

extern "C" int run() {{

    kernel<<<{ngrids}, {nthreads}>>>(d_a, d_b, d_c); 

    isfinished = 1;

    return 0;
}}
"""


class CudaEngine(Engine):

    name = "cuda"

    def __init__(self, workdir):

        super(CudaEngine, self).__init__(workdir)

        compiler = which("nvcc")
        if compiler is None or not os.path.isfile(compiler):
            raise Exception("nvcc is not found")

        self.compiler = os.path.realpath(compiler)

        # TODO: compile and load runtime library wrapper per compiler

        self.rootdir = os.path.join(os.path.dirname(self.compiler), "..")

        self.incdir = os.path.join(self.rootdir, "include")
        if not os.path.isdir(self.incdir):
            raise Exception("Can not find include directory")

        self.libdir = os.path.join(self.rootdir, "lib64")
        if not os.path.isdir(self.libdir):
            self.libdir = os.path.join(self.rootdir, "lib")

            if not os.path.isdir(self.libdir):
                raise Exception("Can not find library directory")

        #self.libcudart = load_library("libcudart", self.libdir)

    def gencode(self, nteams, nmembers, inargs, outargs, order):
        
        # generate source code

        # {dvardefs} {dvarcopyins} {dvarcopyouts} {devcodebody} {ngrids} {nthreads}
        ng = str(nteams)
        nt = str(nmembers)
        dcb = "\n".join(order.sections["cuda"][2])

        innames, outnames = order.get_argnames()

        assert len(innames) == len(inargs), "The number of input arguments mismatches."
        assert len(outnames) == len(outargs), "The number of input arguments mismatches."

        dvd = ""        
        for aname, (arg, attr) in zip(innames+outnames, inargs+outargs):
            self.argmap[id(arg)] = aname
            dvd += "double * h_%s;\n" % aname
            dvd += "int h_%s_size;\n" % aname
            dvd += "__device__ double * d_%s;\n" % aname
            dvd += "__device__ int * d_%s_size;\n" % aname

        dvci = ""        
        dvco = ""        

        code = code_template.format(dvardefs=dvd, dvarcopyins=dvci, dvarcopyouts=dvco,
            devcodebody=dcb, ngrids=ng, nthreads=nt)


        codepath = os.path.join(self.workdir, "test.cu")
        with open(codepath, "w") as f:
            f.write(code)

        import pdb; pdb.set_trace()
        # compile
        opts = ""

        outpath = os.path.join(self.workdir, "mylib.so")

        # generate shared library
        cmdopts = {"nvcc": self.compiler, "opts": opts, "path": codepath,
                    "defaults": "--compiler-options '-fPIC' -o %s --shared" % outpath
                }

        cmd = "{nvcc} {opts} {defaults} {path}".format(**cmdopts)
        out = subp.run(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE, check=False)

        #import pdb; pdb.set_trace()

        if out.returncode  != 0:
            print(out.stderr)
            sys.exit(out.returncode)

        head, tail = os.path.split(outpath)
        base, ext = os.path.splitext(tail)

        #array_1d_double = ndpointer(dtype=double, ndim=1, flags='CONTIGUOUS')

        # load the library, using numpy mechanisms
        self.kernel = load_library(base, head)

        return self.kernel

        # setup the return types and argument types
        #libkernel.run.restype = None
        #libkernel.run.argtypes = [array_1d_double, array_1d_double, c_int]

        # launch cuda program
        #th = Thread(target=self.sharedlib.run)
        #th.start()



    def h2dcopy(self, inargs, outargs):

        for arg, attr in inargs+outargs:
            #np.ascontiguousarray(x, dtype=np.float32)
            name = self.argmap[id(arg)]
            h2dcopy = getattr(self.kernel, "h2dcopy_%s" % name)
            h2dcopy.restype = None
            h2dcopy.argtypes = [ndpointer(c_double), c_size_t]

            h2dcopy(arg, arg.size)

    def d2hcopy(self, outargs):

        for arg, attr in outargs:
            name = self.argmap[id(arg)]
            d2hcopy = getattr(self.kernel, "d2hcopy_%s" % name)
            d2hcopy.restype = None
            d2hcopy.argtypes = [ndpointer(c_double), c_size_t]

            d2hcopy(arg, arg.size)

