"""Errand CUDA engine module


"""

import os, sys, abc
import subprocess as subp

from numpy import double
from numpy.ctypeslib import load_library, ndpointer

from errand.engine import Engine
from errand.util import which


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
            raise Exception("Can not find library directory")

        self.libdir = os.path.join(self.rootdir, "lib64")
        if not os.path.isdir(self.libdir):
            raise Exception("Can not find library directory")

        #self.libcudart = load_library("libcudart", self.libdir)

    def gencode(self, nteams, nmembers, inargs, outargs, order):
        
        # generate source code

        code = """
#include <stdio.h>
#include <unistd.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\\n");
}

int isfinished = 0;

extern "C" int isalive() {

    return isfinished;
}

extern "C" int run() {
    cuda_hello<<<1,1>>>(); 

    isfinished = 1;

    return 0;
}
"""
        codepath = os.path.join(self.workdir, "test.cu")
        with open(codepath, "w") as f:
            f.write(code)

        # compile
        opts = ""

        outpath = os.path.join(self.workdir, "mylib.so")

        # generate shared library
        cmdopts = {"nvcc": self.compiler, "opts": opts, "path": codepath,
                    "defaults": "--compiler-options '-fPIC' -o %s --shared" % outpath
                }

        cmd = "{nvcc} {opts} {defaults} {path}".format(**cmdopts)
        out = subp.run(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE, check=False)

        if out.returncode  != 0:
            print(out.stderr)
            sys.exit(out.returncode)

        head, tail = os.path.split(outpath)
        base, ext = os.path.splitext(tail)

        array_1d_double = ndpointer(dtype=double, ndim=1, flags='CONTIGUOUS')

        # load the library, using numpy mechanisms
        return load_library(base, head)

        # setup the return types and argument types
        #libkernel.run.restype = None
        #libkernel.run.argtypes = [array_1d_double, array_1d_double, c_int]

        # launch cuda program
        #th = Thread(target=self.sharedlib.run)
        #th.start()


    def h2dcopy(self, inargs, outargs):
        pass

    def d2hcopy(self, outargs):
        pass
