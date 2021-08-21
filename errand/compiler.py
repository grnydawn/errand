"""Errand compiler module

"""

import os, abc, re

from errand.util import which, shellcmd

re_gcc_version = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d)+")

class Compiler(abc.ABC):
    """Parent class for all compiler classes

"""

    def __init__(self, path):

        self.path = path
        ver = self.get_version()
        self.version = ver if ver else None

    def isavail(self):
        return (self.path is not None and os.path.isfile(self.path) and
                self.version is not None)

    @abc.abstractmethod
    def get_option(self):
        pass

    def get_version(self):
        return shellcmd("%s --version" % self.path).stdout.decode()


class CPP_Compiler(Compiler):

    def __init__(self, path):
        super(CPP_Compiler, self).__init__(path)


class GNU_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("g++")

        super(GNU_CPP_Compiler, self).__init__(path)

    def get_option(self):
        return "-shared -fPIC"

    def get_version(self):
        ver = super(GNU_CPP_Compiler, self).get_version()
        match = re_gcc_version.search(ver)
        if match:
            return  ver


class OpenAcc_GNU_CPP_Compiler(GNU_CPP_Compiler):

    def __init__(self, path=None):

        super(OpenAcc_GNU_CPP_Compiler, self).__init__(path)

    def get_option(self):

        return ("-fopenacc " +
                super(OpenAcc_GNU_CPP_Compiler, self).get_option(self))


class CUDA_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("nvcc")

        super(CUDA_CPP_Compiler, self).__init__(path)

    def get_option(self):
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

        elif engine == "openacc-c++":
            self.clist =  [OpenAcc_GNU_CPP_Compiler()]

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

