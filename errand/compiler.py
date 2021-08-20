"""Errand compiler module

"""

import os, abc

from errand.util import which


class Compiler(abc.ABC):
    """Parent class for all compiler classes

"""

    def __init__(self, path):

        self.path = path

    def isavail(self):
        return self.path is not None and os.path.isfile(self.path)

    @abc.abstractmethod
    def get_option(self):
        pass

class C_Compiler(Compiler):

    def __init__(self, path):
        super(C_Compiler, self).__init__(path)


class CPP_Compiler(C_Compiler):

    def __init__(self, path):

        super(CPP_Compiler, self).__init__(path)


class GNU_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("g++")

        super(GNU_CPP_Compiler, self).__init__(path)

    def get_option(self):
        return "-shared -fPIC"


class CUDA_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("nvcc")

        super(CUDA_CPP_Compiler, self).__init__(path)

    def get_option(self):
        #return "--compiler-options '-fPIC' --shared -std=c++11"
        return "--compiler-options '-fPIC' --shared"

class HIP_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("hipcc")

        super(HIP_CPP_Compiler, self).__init__(path)

    def get_option(self):
        return "-fPIC --shared"


class Compilers(object):

    def __init__(self, engine):

        if engine == "pthread":
            self.clist =  [GNU_CPP_Compiler()]

        elif engine == "cuda":
            self.clist =  [CUDA_CPP_Compiler()]

        elif engine == "hip":
            self.clist =  [HIP_CPP_Compiler()]

        else:
            raise Exception("Compiler for '%s' is not supported." % engine)

    def isavail(self):

        return self.select_one() is not None        

    def select_one(self):

        for comp in self.clist:
            if comp.isavail():
                return comp

    def select_many(self):

        comps = []

        for comp in self.clist:
            if comp.isavail():
                comps.append(comp)

        return comps

