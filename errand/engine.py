"""Errand accellerator engine module


"""

import os, abc

from numpy.ctypeslib import load_library
from errand.util import which

class Engine(object):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    @abc.abstractmethod
    def devmalloc(self, size):
        pass


class CudaEngine(Engine):

    name = "cuda"

    def __init__(self):

        compiler = which("nvcc")
        if compiler is None or not os.path.isfile(compiler):
            raise Exception("nvcc is not found")

        self.compiler = os.path.realpath(compiler)

        # TODO: compile and load runtime library wrapper per compiler

#        self.rootdir = os.path.join(os.path.dirname(self.compiler), "..")
#
#        self.incdir = os.path.join(self.rootdir, "include")
#        if not os.path.isdir(self.incdir):
#            raise Exception("Can not find include directory")
#
#        self.libdir = os.path.join(self.rootdir, "lib64")
#        if not os.path.isdir(self.libdir):
#            raise Exception("Can not find library directory")
#
#        self.libdir = os.path.join(self.rootdir, "lib64")
#        if not os.path.isdir(self.libdir):
#            raise Exception("Can not find library directory")
#
#        self.libcudart = load_library("libcudart", self.libdir)


    def devmalloc(self, size):

        # (ctypes.c_ulong*3)()
        #a_p = a.ctypes.data_as(POINTER(c_float))

        import pdb; pdb.set_trace()
        pass

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

    def devmalloc(self, size):
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
