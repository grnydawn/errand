"""Errand CUDA engine module


"""

import os, sys, abc
import subprocess as subp

from numpy import double
from numpy.ctypeslib import load_library, ndpointer
from ctypes import c_double, c_size_t

from errand.engine import Engine
from errand.util import which


dtypemap = {
    "float64": "double"
}

code_template = """
//#include <stdio.h>
//#include <unistd.h>

int isfinished = 0;

//using namespace std;

// TODO: prepare all possible type/dim combinations
// dim: 0, 1,2,3,4,5
// type: int, float, char, boolean

{dvarstructs}

{dvardefs}

{dvarcopyins}

{dvarcopyouts}

__global__ void _kernel({devcodeargs}){{
    {devcodebody}
}}

extern "C" int isalive() {{

    return isfinished;
}}

extern "C" int run() {{

    _kernel<<<{ngrids}, {nthreads}>>>({hostcallargs}); 

    isfinished = 1;

    return 0;
}}
"""


class CudaEngine(Engine):

    def __init__(self, workdir):

        super(CudaEngine, self).__init__(workdir)

        compiler = which("nvcc")
        if compiler is None or not self.isavail():
            raise Exception("nvcc is not found")

        self.compiler = os.path.realpath(compiler)

    @classmethod
    def isavail(self):

        compiler = which("nvcc")
        if compiler is None or not os.path.isfile(compiler):
            return False

        rootdir = os.path.join(os.path.dirname(compiler), "..")

        incdir = os.path.join(rootdir, "include")
        if not os.path.isdir(incdir):
            return False

        libdir = os.path.join(rootdir, "lib64")
        if not os.path.isdir(libdir):
            libdir = os.path.join(rootdir, "lib")

            if not os.path.isdir(libdir):
                return False

        return True

    def gencode(self, nteams, nmembers, inargs, outargs, order):
        
        # generate source code

        # {dvardefs} {dvarcopyins} {dvarcopyouts} {devcodebody} {ngrids} {nthreads}
        ng = str(nteams)
        nt = str(nmembers)
        dcb = "\n".join(order.sections["cuda"][2])

        innames, outnames = order.get_argnames()

        assert len(innames) == len(inargs), "The number of input arguments mismatches."
        assert len(outnames) == len(outargs), "The number of input arguments mismatches."

        dvs = {}
        dvd = ""        
        dvci = ""        
        dca = []
        hca = []
        for aname, (arg, attr) in zip(innames+outnames, inargs+outargs):
            self.argmap[id(arg)] = aname

            dtname = dtypemap[arg.dtype.name]

            if dtname in dvs:
                dvsd = dvs[dtname]

            else:
                dvsd = {}
                dvs[dtname] = dvsd
                
            ndim = str(arg.ndim)
            if ndim not in dvsd:
                dvsdn = ""

                dvsdn += "struct %s_dim%s {\n" % (dtname, ndim)
                dvsdn += "    %s * data;\n" % dtname
                dvsdn += "    int * _size;\n"
                dvsdn += "    __device__ int size() {;\n"
                dvsdn += "        return * _size;\n"
                dvsdn += "    }\n"
                dvsdn += "};\n"

                dvsd[ndim] = dvsdn

            dvd += "double * h_%s;\n" % aname
            dvd += "__device__ %s_dim%s d_%s;\n" % (dtname, ndim, aname)

            dvci += "extern \"C\" void h2dcopy_%s(void * data, int size) {\n" % aname
            dvci += "    h_%s = (double *) data;\n" % aname
            dvci += "    cudaMalloc((void **)&d_%s.data, size * sizeof(double));\n" % aname
            dvci += "    cudaMalloc((void **)&d_%s._size, sizeof(int));\n" % aname
            dvci += "    cudaMemcpy(d_%s.data, h_%s, size * sizeof(double), cudaMemcpyHostToDevice);\n" % (aname, aname)
            dvci += "    cudaMemcpy(d_%s._size, &size, sizeof(int), cudaMemcpyHostToDevice);\n" % aname
            dvci += "}\n"

            dca.append("%s_dim%s %s" % (dtname, ndim, aname))

            hca.append("d_%s" % aname)

        dvco = ""
        for aname, (arg, attr) in zip(outnames, outargs):
            dvco += "extern \"C\" void d2hcopy_%s(void * data, int size) {\n" % aname
            dvco += "    cudaMemcpy(h_%s, d_%s.data, size * sizeof(double), cudaMemcpyDeviceToHost);\n" % (aname, aname)
            dvco += "    data = (void *) h_%s;\n" % aname
            dvco += "}\n"

        dvs_str = "\n".join([y for x in dvs.values() for y in x.values()])

        code = code_template.format(dvardefs=dvd, dvarcopyins=dvci, dvarcopyouts=dvco,
            devcodebody=dcb, devcodeargs=", ".join(dca), hostcallargs=", ".join(hca),
            dvarstructs=dvs_str, ngrids=ng, nthreads=nt)

        codepath = os.path.join(self.workdir, "test.cu")
        with open(codepath, "w") as f:
            f.write(code)

        #import pdb; pdb.set_trace()
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

