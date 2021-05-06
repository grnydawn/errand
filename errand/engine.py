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

    def kernel_launch(self, esf, config, workdir, *vargs):

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

        import pdb; pdb.set_trace()

        # gen code name
        path = os.path.join(self.workdir, "code.cu")

        # write code
        with open(path, "w") as f:
            f.write(sig + "\n")
            f.write("{" + "\n")
            f.write(body + "\n")
            f.write("}")

        code["path"] = path

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

        try:
            cmd = "{nvcc} {opts} {defaults} {path}".format(cmdopts)
            stdout = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)

        except Exception as err:
            print(err.stdout)
            sys.exit(err.returncode)

        return nvcc

