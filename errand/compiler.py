"""Errand compiler module

"""

import os, abc, re

from errand.util import which, shellcmd


class Compiler(abc.ABC):
    """Parent class for all compiler classes

"""

    def __init__(self, path, flags):

        self.path = path
        self.flags = flags
        self.version = None

    def isavail(self):

        if self.version is None:
            self.set_version(self.get_version())

        return (self.path is not None and os.path.isfile(self.path) and
                self.version is not None)

    def set_version(self, version):

        if version and self.check_version(version):
            self.version = version

    @abc.abstractmethod
    def get_option(self):
        return " ".join(self.flags) if self.flags else ""

    def get_version(self):
        ver = shellcmd("%s --version" % self.path).stdout.decode()
        return ver if ver else None

    @abc.abstractmethod
    def check_version(self, version):
        return False


class CPP_Compiler(Compiler):

    def __init__(self, path, flags):
        super(CPP_Compiler, self).__init__(path, flags)


class GNU_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("g++")

        super(GNU_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared -fPIC " + super(GNU_CPP_Compiler, self).get_option()

    def check_version(self, version):

        return version.startswith("g++ (GCC)")


class AmdClang_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("CC")

        super(AmdClang_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(AmdClang_CPP_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("clang version")


class CrayClang_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("CC")

        super(CrayClang_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(CrayClang_CPP_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("Cray clang version")


class IbmXl_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("xlc++")

        super(IbmXl_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(IbmXl_CPP_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("IBM XL C/C++")


class Pthread_GNU_CPP_Compiler(GNU_CPP_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_GNU_CPP_Compiler, self).get_option()


class Pthread_CrayClang_CPP_Compiler(CrayClang_CPP_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_CrayClang_CPP_Compiler, self).get_option()


class Pthread_AmdClang_CPP_Compiler(AmdClang_CPP_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_AmdClang_CPP_Compiler, self).get_option()


class OpenAcc_GNU_CPP_Compiler(Pthread_GNU_CPP_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_GNU_CPP_Compiler, self).__init__(path, flags)

        self.version = self.get_version()

    def get_option(self):

        return ("-fopenacc " +
                super(OpenAcc_GNU_CPP_Compiler, self).get_option())

    def check_version(self, version):

        pat = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d)+")

        match = pat.search(version)

        if not match:
            return False

        return int(match.group("major")) >= 10


class CUDA_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("nvcc")

        super(CUDA_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):
        return ("--compiler-options '-fPIC' --shared " +
                super(CUDA_CPP_Compiler, self).get_option())

    def check_version(self, version):
        return version.startswith("nvcc: NVIDIA")


class HIP_CPP_Compiler(CPP_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("hipcc")

        super(HIP_CPP_Compiler, self).__init__(path, flags)

    def get_option(self):

        return ("-fPIC --shared " +
                super(HIP_CPP_Compiler, self).get_option())

    def check_version(self, version):
        return version.startswith("HIP version")



class Compilers(object):

    def __init__(self, backend, compile):

        self.clist = []

        clist = []

        if backend == "pthread":
            clist =  [Pthread_GNU_CPP_Compiler, Pthread_CrayClang_CPP_Compiler,
                      Pthread_AmdClang_CPP_Compiler]

        elif backend == "cuda":
            clist =  [CUDA_CPP_Compiler]

        elif backend == "hip":
            clist =  [HIP_CPP_Compiler]

        elif backend == "openacc-c++":
            clist =  [OpenAcc_GNU_CPP_Compiler]

        else:
            raise Exception("Compiler for '%s' is not supported." % backend)

        for cls in clist:
            try:
                if compile:
                    path = which(compile[0])
                    if path:
                        self.clist.append(cls(path, compile[1:]))

                else:
                    self.clist.append(cls(None, None))

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

