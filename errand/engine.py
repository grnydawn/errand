"""Errand engine module"""

import os, sys, abc, string
import subprocess as subp

Object = abc.ABCMeta("Object", (object,), {})

LEN_FILENAME = 16

def _which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p


class Engine(Object):

    def __init__(self, compiler, workdir):

        self.compiler = compiler
        self.workdir = workdir

    def kernel_launch(self, orderpads, esf, config, argnames, *vargs):

        # generate cuda source code
        code = self.gencode(orderpads, esf, config, argnames, *vargs)

        # compile the code to build so file
        obj = self.genobj(code)

        # load the so file
        accel = self.loadobj(obj)

        # launch cuda kernel
        accel(**vargs)

    def loadobj(self, obj):

        head, tail = os.path.split(obj) 
        base, ext = os.path.splitext(tail) 

        sys.path.insert(0, os.path.abspath(os.path.realpath(head)))
        m = __import__(base)
        sys.path.pop(0)

        return m

    @abc.abstractmethod
    def gencode(self, esf, *vargs):
        pass

    @abc.abstractmethod
    def genobj(self, code):
        pass

    def findcompiler(self):
        pass


class CUDAEngine(Engine):

    def vartype2str(self, vartype):
        import pdb; pdb.set_trace()

    def gensig(self, orderpads, argnames, sig, attrs, sbody):

        FNAME = "Errand_Cuda_Function"
        out = ""

        _args = []

        if sig:

            # parse sig
            pos = sig.find("->")

            # TODO: add cuda specific C type indicaters

            if pos >= 0:
                _args = ([x.strip() for x in sig[:pos].split(",")] + ["->"] +
                        [x.strip() for x in sig[pos+2:].split(",")])

            else:
                _args = [x.strip() for x in sig.split(",")]

        else:
            _args = argnames 

        # TODO: get the index of arg and use the index to match with real variable
        # transform arg to cuda arg
        args = []
        isInput = True
        for _a in _args:

            if _a == "->":
                isInput = False
                continue

            vartypestr = self.vartype2str(orderpads.get_vartype(_a, isInput))
            args.append(vartypestr + " " + _a)

        return "__global__ void %s(%s)" % (FNAME, ", ".join(args))

    def genbody(self, orderpads, config, opts, attrs, body):

        # TODO: allocation, copy to device

        compute = "\n".join(body)

        # TODO: deallocation, copy to host

    def gencode(self, orderpads, esf, config, argnames, *vargs):

        code = {}

        raw_sig = esf.get_signature()
        raw_body = esf.get_section("cuda")

        orderpads.load_arguments(*vargs)
        sig = self.gensig(orderpads, argnames, *raw_sig)
        body = self.genbody(orderpads, config, *raw_body)

        # gen code name
        path = os.path.join(self.workdir, "code.cu")

        # TODO: add shared library boiler-plate code

        # write code
        with open(path, "w") as f:
            f.write(sig + "\n")
            f.write("{" + "\n")
            f.write(body + "\n")
            f.write("}")

        code["path"] = path

        return code

    def genobj(self, code):

        # TODO: find nvcc

        nvcc = None
        # find compiler
        # 1. from function call argument
        # 2. from esf file
        # 3. from env variable
        # 4. from nvcc command
        # 5. from well known locations
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

        # collect compiler options

        opts = ""

        if "compiler_options" in code:
            opts = code["compiler_options"]

        elif "ERRAND_COMPILER_OPTION" in os.environ:
            opts = os.environ["ERRAND_COMPILER_OPTIONS"]

        # generate shared library
        cmdopts = {"nvcc": nvcc, "opts": opts, "path": code["path"],
                    "defaults": "--compiler-options '-fPIC' -o mylib.so --shared"
                }

        cmd = "{nvcc} {opts} {defaults} {path}".format(**cmdopts)
        #stdout = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
        out = subp.run(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE, check=False)

        import pdb ;pdb.set_trace()
        if out.returncode  != 0:
            print(out.stderr)
            sys.exit(out.returncode)

        return nvcc

