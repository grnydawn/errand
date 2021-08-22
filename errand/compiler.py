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
        self.version = None

    def isavail(self):

        if self.version is None:
            self.version = self.get_version()

        return (self.path is not None and os.path.isfile(self.path) and
                self.version is not None)

    @abc.abstractmethod
    def get_option(self):
        return ""

    def get_version(self):

        ver = shellcmd("%s --version" % self.path).stdout.decode()

        return ver if ver else None



class CPP_Compiler(Compiler):

    def __init__(self, path):
        super(CPP_Compiler, self).__init__(path)


class GNU_CPP_Compiler(CPP_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("g++")

        super(GNU_CPP_Compiler, self).__init__(path)

    def get_option(self):
        return "-shared -fPIC " + super(GNU_CPP_Compiler, self).get_option()


class Pthread_GNU_CPP_Compiler(GNU_CPP_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_GNU_CPP_Compiler, self).get_option()


class OpenAcc_GNU_CPP_Compiler(Pthread_GNU_CPP_Compiler):

    def __init__(self, path=None):

        super(OpenAcc_GNU_CPP_Compiler, self).__init__(path)

        self.version = self.get_version()

        match = re_gcc_version.search(self.version)

        if not match:
            raise Exception("Can not parse GCC version string: %s" %
                                str(self.version)) 
        if int(match.group("major")) < 10:
            raise Exception("Gcc version should be at least 10: %s" %
                                str(self.version)) 

    def get_option(self):

        return ("-fopenacc " +
                super(OpenAcc_GNU_CPP_Compiler, self).get_option())


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

        self.clist = []

        clist = []

        if engine == "pthread":
            clist =  [Pthread_GNU_CPP_Compiler]

        elif engine == "cuda":
            clist =  [CUDA_CPP_Compiler]

        elif engine == "hip":
            clist =  [HIP_CPP_Compiler]

        elif engine == "openacc-c++":
            clist =  [OpenAcc_GNU_CPP_Compiler]

        else:
            raise Exception("Compiler for '%s' is not supported." % engine)

        for cls in clist:
            try:
                self.clist.append(cls())

            except Exception as err:
                pass

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

