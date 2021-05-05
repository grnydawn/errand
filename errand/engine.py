"""Errand engine module"""

import os, sys, abc

Object = abc.ABCMeta("Object", (object,), {})

def _which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p


class Engine(Object):

    def __init__(self):

        self.compiler = None

    def kernel_launch(self, esf, config, *vargs):

        # generate cuda source code
        code = self.gencode(esf, *vargs)

        # compile the code to build so file
        obj = self.genobj(code)

        # load the so file
        accel = self.loadobj(obj)

        # launch cuda kernel
        accel(config, **vargs)

    def loadobj(self, obj):
        # TODO: implement this
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

    def gencode(self, esf, *vargs):

        code = {}

        sig = esf.get_signature()
        body = esf.get_section("cuda")

        code["code"] = (sig, body)

        return code

    def genobj(self, code):

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
            os.environ["ERRAND_COMPILER"]
            
        else:
            _nvcc = _which("nvcc")
            if _nvcc:
                nvcc = _nvcc

            else:
                _nvcc = self.findcompiler()
                if _nvcc:
                    nvcc = _nvcc

        if nvcc is None:
            raise Exception("Can not find CUDA compiler.")

        # collect compiler options

        # generate shared library

        return ""

