"""Errand accelerator engine module


"""

import os, sys, abc

import subprocess as subp
from numpy.ctypeslib import load_library
from errand.util import which

class Engine(abc.ABC):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    @abc.abstractmethod
    def gen_code(self, order, workdir):
        pass

    @abc.abstractmethod
    def gen_sharedlib(self, code, workdir):
        pass

class CudaEngine(Engine):

    name = "cuda"

    def __init__(self):

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

        self.libcudart = load_library("libcudart", self.libdir)


    def gen_code(self, order, workdir):

        code = """
#include <stdio.h>
#include <unistd.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\\n");
}

int x = 0;

extern "C" int stop() {
    x = 1;

    return 0;
}

extern "C" int run() {
    cuda_hello<<<1,1>>>(); 

    while(x == 0) {
        printf("Hello World from CPU!\\n");
        usleep(1000000);
    }

    return 0;
}
"""
        path = os.path.join(workdir, "test.cu")
        with open(path, "w") as f:
            f.write(code)
        
        return path

    def gen_sharedlib(self, codepath, workdir):

        if self.compiler:
            nvcc = self.compiler

        elif "compiler" in code:
            nvcc = code["compiler"]

        elif "ERRAND_COMPILER" in os.environ:
            nvcc = os.environ["ERRAND_COMPILER"]

        else:
            nvcc = _which("nvcc")

            if nvcc is None:
                nvcc = self.findcompiler()

        if nvcc is None:
            raise Exception("Can not find CUDA compiler.")

        opts = ""

        outpath = os.path.join(workdir, "mylib.so")

        # generate shared library
        cmdopts = {"nvcc": nvcc, "opts": opts, "path": codepath,
                    "defaults": "--compiler-options '-fPIC' -o %s --shared" % outpath
                }

        cmd = "{nvcc} {opts} {defaults} {path}".format(**cmdopts)
        out = subp.run(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE, check=False)

        if out.returncode  != 0:
            print(out.stderr)
            sys.exit(out.returncode)

        return outpath


    #def devmalloc(self, size):

        # (ctypes.c_ulong*3)()
        #a_p = a.ctypes.data_as(POINTER(c_float))

    #    import pdb; pdb.set_trace()
    #    pass

class HipEngine(Engine):

    name = "hip"

    def __init__(self):

        compiler = which("hipcc")

        if compiler is None or not os.path.isfile(compiler):
            raise Exception("hipcc is not found")

        self.compiler = os.path.realpath(compiler)

        self.rootdir = os.path.join(os.path.dirname(self.compiler), "..")

        self.incdir = os.path.join(self.rootdir, "include")
        if not os.path.isdir(self.incdir):
            raise Exception("Can not find include directory")

        self.libdir = os.path.join(self.rootdir, "lib")
        if not os.path.isdir(self.libdir):
            self.libdir = os.path.join(self.rootdir, "lib64")

            if not os.path.isdir(self.libdir):
                raise Exception("Can not find library directory")

    def gen_code(self, order, workdir):
        pass

    def gen_sharedlib(self, code, workdir):
        pass


def select_engine(engine, order):

    if isinstance(engine, Engine):
        return engine

    if isinstance(engine, str):
        if engine == "cuda":
            return CudaEngine()

        elif engine == "hip":
            return HipEngine()

        else:
            raise Exception("Not supported engine type: %s" % engine)

    elif order:
        pass
 
    # TODO: auto-select engine from sysinfo
