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

struct DoubleDim1 {{
    double * data;
    int * _size;

    __device__ int size() {{
        return *_size;
    }}
}};

{dvardefs}

{dvarcopyins}

{dvarcopyouts}

__global__ void kernel({devcodeargs}){{
    {devcodebody}
}}

extern "C" int isalive() {{

    return isfinished;
}}

extern "C" int run() {{

    kernel<<<{ngrids}, {nthreads}>>>({hostcallargs}); 

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
        dvci = ""        
        dca = []
        hca = []
        for aname, (arg, attr) in zip(innames+outnames, inargs+outargs):
            self.argmap[id(arg)] = aname
            dvd += "double * h_%s;\n" % aname
            dvd += "int h_%s_size;\n" % aname
            dvd += "__device__ DoubleDim1 d_%s;\n" % aname

            dvci += "extern \"C\" void h2dcopy_%s(void * data, int size) {\n" % aname
            dvci += "    h_%s = (double *) data;\n" % aname
            dvci += "    h_%s_size = size;\n" % aname
            dvci += "    cudaMalloc((void **)&d_%s.data, size * sizeof(double));\n" % aname
            dvci += "    cudaMalloc((void **)&d_%s._size, sizeof(int));\n" % aname
            dvci += "    cudaMemcpy(d_%s.data, h_%s, size * sizeof(double), cudaMemcpyHostToDevice);\n" % (aname, aname)
            dvci += "    cudaMemcpy(d_%s._size, &h_%s_size, sizeof(int), cudaMemcpyHostToDevice);\n" % (aname, aname)
            dvci += "}\n"

            dca.append("DoubleDim1 %s" % aname)

            hca.append("d_%s" % aname)

        dvco = ""
        for aname, (arg, attr) in zip(outnames, outargs):
            dvco += "extern \"C\" void d2hcopy_c(void * data, int size) {\n"
            dvco += "    cudaMemcpy(h_%s, d_%s.data, size * sizeof(double), cudaMemcpyDeviceToHost);\n" % (aname, aname)
            dvco += "    data = (void *) h_%s;\n" % aname
            dvco += "}\n"


        code = code_template.format(dvardefs=dvd, dvarcopyins=dvci, dvarcopyouts=dvco,
            devcodebody=dcb, devcodeargs=", ".join(dca), hostcallargs=", ".join(hca),
            ngrids=ng, nthreads=nt)

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

        # load the library, using numpy mechanisms
        self.kernel = load_library(base, head)

        return self.kernel

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

