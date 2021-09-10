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
        return ver.strip() if ver else None

    @abc.abstractmethod
    def check_version(self, version):
        return False


class Cpp_Compiler(Compiler):

    def __init__(self, path, flags):
        super(Cpp_Compiler, self).__init__(path, flags)


class Gnu_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("g++")

        super(Gnu_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared -fPIC " + super(Gnu_Cpp_Compiler, self).get_option()

    def check_version(self, version):

        return version.startswith("g++ (GCC)")


class AmdClang_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("CC")

        super(AmdClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(AmdClang_Cpp_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("clang version") and "roc" in version


class Pgi_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("pgc++")

        super(Pgi_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(Pgi_Cpp_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("pgc++") and "PGI" in version


class CrayClang_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("CC")

        super(CrayClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(CrayClang_Cpp_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("Cray clang version")


class IbmXl_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("xlc++")

        super(IbmXl_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return "-shared " + super(IbmXl_Cpp_Compiler, self).get_option()

    def check_version(self, version):
        return version.startswith("IBM XL C/C++")


class Pthread_Gnu_Cpp_Compiler(Gnu_Cpp_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_Gnu_Cpp_Compiler, self).get_option()


class Pthread_CrayClang_Cpp_Compiler(CrayClang_Cpp_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_CrayClang_Cpp_Compiler, self).get_option()


class Pthread_AmdClang_Cpp_Compiler(AmdClang_Cpp_Compiler):

    def get_option(self):
        return "-pthread " + super(Pthread_AmdClang_Cpp_Compiler, self).get_option()


class Pthread_Pgi_Cpp_Compiler(Pgi_Cpp_Compiler):

    def get_option(self):
        return "-lpthread " + super(Pthread_Pgi_Cpp_Compiler, self).get_option()


class OpenAcc_Gnu_Cpp_Compiler(Pthread_Gnu_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_Gnu_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):

        return ("-fopenacc " +
                super(OpenAcc_Gnu_Cpp_Compiler, self).get_option())

    def check_version(self, version):

        pat = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d)+")

        match = pat.search(version)

        if not match:
            return False

        return int(match.group("major")) >= 10


class OpenAcc_CrayClang_Cpp_Compiler(Pthread_CrayClang_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_CrayClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):

        return ("-h pragma=acc " +
                super(OpenAcc_CrayClang_Cpp_Compiler, self).get_option())


class OpenAcc_Pgi_Cpp_Compiler(Pthread_Pgi_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_Pgi_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return ("-acc " +
                super(OpenAcc_Pgi_Cpp_Compiler, self).get_option())


class Cuda_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("nvcc")

        super(Cuda_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):
        return ("--compiler-options '-fPIC' --shared " +
                super(Cuda_Cpp_Compiler, self).get_option())

    def check_version(self, version):
        return version.startswith("nvcc: NVIDIA")


class Hip_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("hipcc")

        super(Hip_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self):

        return ("-fPIC --shared " +
                super(Hip_Cpp_Compiler, self).get_option())

    def check_version(self, version):
        return version.startswith("HIP version")



class Compilers(object):

    def __init__(self, backend, compile):

        self.clist = []

        clist = []

        if backend in ("pthread", "c++"):
            clist =  [Pthread_Gnu_Cpp_Compiler, Pthread_CrayClang_Cpp_Compiler,
                      Pthread_AmdClang_Cpp_Compiler, Pthread_Pgi_Cpp_Compiler]

        elif backend == "cuda":
            clist =  [Cuda_Cpp_Compiler]

        elif backend == "hip":
            clist =  [Hip_Cpp_Compiler]

        elif backend == "openacc-c++":
            clist =  [OpenAcc_Gnu_Cpp_Compiler, OpenAcc_CrayClang_Cpp_Compiler,
                      OpenAcc_Pgi_Cpp_Compiler]
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

