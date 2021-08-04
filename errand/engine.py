"""Errand engine module


"""

import os, sys, abc

from numpy.ctypeslib import load_library

from errand.util import which


class Engine(abc.ABC):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    def __init__(self, workdir):
        self.workdir = workdir


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

        self.libcudart = load_library("libcudart", self.libdir)

def select_engine(engine, order):

    if isinstance(engine, Engine):
        return engine.__class__

    if isinstance(engine, str):
        if engine == "cuda":
            return CudaEngine

        elif engine == "hip":
            return HipEngine

        else:
            raise Exception("Not supported engine type: %s" % engine)

    elif order:
        raise Exception("Engine-selection from order is not supported")

    # TODO: auto-select engine from sysinfo
