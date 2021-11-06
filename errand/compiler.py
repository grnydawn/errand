"""Errand compiler module

"""

import os, sys, abc, re

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
    def get_option(self, **kwargs):

        linker = kwargs.pop("linker", True)

        opt = " ".join(self.flags) if self.flags else ""

        if linker is False:
            opt += " -c "

        return opt

    def get_version(self):
        ver = shellcmd("%s --version" % self.path).stdout.decode()
        return ver.strip() if ver else None

    @abc.abstractmethod
    def check_version(self, version):
        return False


class Cpp_Compiler(Compiler):

    def __init__(self, path, flags):
        super(Cpp_Compiler, self).__init__(path, flags)


class Fortran_Compiler(Compiler):

    def __init__(self, path, flags):
        super(Fortran_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        opt = " "

        moddir = kwargs.pop("moddir", None)
        if moddir:
            opt = "-J %s " % moddir

        return opt + super(Fortran_Compiler, self).get_option(**kwargs)


class AppleClang_Cpp_Compiler(Cpp_Compiler):

    libext = "dylib"

    def __init__(self, path, flags):

        if path is None:
            path = which("clang++")

        super(AppleClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-dynamiclib -fPIC " + super(AppleClang_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("Apple clang version")


class Gnu_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("g++")

        super(Gnu_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-shared -fPIC " + super(Gnu_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):

        return version.startswith("g++ (GCC)")


class AmdClang_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("clang")

        super(AmdClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-shared " + super(AmdClang_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("clang version") and "roc" in version


class Pgi_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("pgc++")

        super(Pgi_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-shared " + super(Pgi_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("pgc++") and "PGI" in version


class CrayClang_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("CC")

        if path is None:
            path = which("clang++")

        if path is None:
            path = which("crayCC")

        super(CrayClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-shared " + super(CrayClang_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("Cray clang version")


class IbmXl_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("xlc++")

        super(IbmXl_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return "-shared " + super(IbmXl_Cpp_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("IBM XL C/C++")


class Pthread_Gnu_Cpp_Compiler(Gnu_Cpp_Compiler):

    def get_option(self, **kwargs):
        return "-pthread " + super(Pthread_Gnu_Cpp_Compiler, self).get_option(**kwargs)


class Pthread_CrayClang_Cpp_Compiler(CrayClang_Cpp_Compiler):

    def get_option(self, **kwargs):
        return "-pthread " + super(Pthread_CrayClang_Cpp_Compiler, self).get_option(**kwargs)


class Pthread_AmdClang_Cpp_Compiler(AmdClang_Cpp_Compiler):

    def get_option(self, **kwargs):
        return "-pthread " + super(Pthread_AmdClang_Cpp_Compiler, self).get_option(**kwargs)


class Pthread_Pgi_Cpp_Compiler(Pgi_Cpp_Compiler):

    def get_option(self, **kwargs):
        return "-lpthread " + super(Pthread_Pgi_Cpp_Compiler, self).get_option(**kwargs)


class Pthread_AppleClang_Cpp_Compiler(AppleClang_Cpp_Compiler):

    def get_option(self, **kwargs):
        return "-lpthread " + super(Pthread_AppleClang_Cpp_Compiler,
                    self).get_option(**kwargs)


class OpenAcc_Gnu_Cpp_Compiler(Pthread_Gnu_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_Gnu_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        return ("-fopenacc " +
                super(OpenAcc_Gnu_Cpp_Compiler, self).get_option(**kwargs))

    def check_version(self, version):

        pat = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d)+")

        match = pat.search(version)

        if not match:
            return False

        return int(match.group("major")) >= 10


class OpenAcc_CrayClang_Cpp_Compiler(Pthread_CrayClang_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_CrayClang_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        return ("-h pragma=acc " +
                super(OpenAcc_CrayClang_Cpp_Compiler, self).get_option(**kwargs))


class OpenAcc_Pgi_Cpp_Compiler(Pthread_Pgi_Cpp_Compiler):

    def __init__(self, path, flags):

        super(OpenAcc_Pgi_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return ("-acc " +
                super(OpenAcc_Pgi_Cpp_Compiler, self).get_option(**kwargs))


class Cuda_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("nvcc")

        super(Cuda_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):
        return ("--compiler-options '-fPIC' --shared " +
                super(Cuda_Cpp_Compiler, self).get_option(**kwargs))

    def check_version(self, version):
        return version.startswith("nvcc: NVIDIA")


class Hip_Cpp_Compiler(Cpp_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("hipcc")

        super(Hip_Cpp_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        return ("-fPIC --shared " +
                super(Hip_Cpp_Compiler, self).get_option(**kwargs))

    def check_version(self, version):
        return version.startswith("HIP version")


class Gnu_Fortran_Compiler(Fortran_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("gfortran")

        super(Gnu_Fortran_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        opt = " "

        return "-shared -fPIC " + opt + super(Gnu_Fortran_Compiler,
            self).get_option(**kwargs)

    def check_version(self, version):

        return version.startswith("GNU Fortran")


class AmdFlang_Fortran_Compiler(Fortran_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("flang")

        super(AmdFlang_Fortran_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        opt = " "

        return "-shared " + opt + super(AmdFlang_Fortran_Compiler,
            self).get_option(**kwargs)

    def check_version(self, version):

        return version.startswith("flang-new version") and "roc" in version


class Cray_Fortran_Compiler(Fortran_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("ftn")

        if path is None:
            path = which("crayftn")

        super(Cray_Fortran_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        opt = " "

        return "-shared " + opt + super(Cray_Fortran_Compiler,
            self).get_option(**kwargs)

    def check_version(self, version):

        return version.startswith("Cray Fortran")


class AppleGnu_Fortran_Compiler(Gnu_Fortran_Compiler):

    libext = "dylib"

    def check_version(self, version):

        return sys.platform == "darwin" and super(AppleGnu_Fortran_Compiler,
                self).check_version(version)


class IbmXl_Fortran_Compiler(Fortran_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("xlf2008_r")

        if path is None:
            path = which("xlf2008")

        if path is None:
            path = which("xlf2003_r")

        if path is None:
            path = which("xlf2003")

        if path is None:
            path = which("xlf95_r")

        if path is None:
            path = which("xlf95")

        if path is None:
            path = which("xlf90_r")

        if path is None:
            path = which("xlf90")

        super(IbmXl_Fortran_Compiler, self).__init__(path, flags)

    def get_version(self):
        ver = shellcmd("%s -qversion" % self.path).stdout.decode()
        return ver.strip() if ver else None

    def get_option(self, **kwargs):

        opt = " "

        moddir = kwargs.pop("moddir", None)
        if moddir:
            opt = "-qmoddir=%s " % moddir

        return "-qmkshrobj " + opt + super(IbmXl_Fortran_Compiler,
            self).get_option(**kwargs)

    def check_version(self, version):

        return version.startswith("IBM XL Fortran")


class Pgi_Fortran_Compiler(Fortran_Compiler):

    def __init__(self, path, flags):

        if path is None:
            path = which("pgfortran")

        super(Pgi_Fortran_Compiler, self).__init__(path, flags)

    def get_option(self, **kwargs):

        opt = " "

        moddir = kwargs.pop("moddir", None)
        if moddir:
            opt = "-module %s " % moddir

        return "-shared -fpic " + opt + super(Pgi_Fortran_Compiler, self).get_option(**kwargs)

    def check_version(self, version):
        return version.startswith("pgfortran") and "PGI" in version


class Compilers(object):

    def __init__(self, backend, compile):

        self.clist = []

        clist = []

        if backend in ("pthread", "c++"):
            clist =  [Pthread_Gnu_Cpp_Compiler, Pthread_CrayClang_Cpp_Compiler,
                      Pthread_AmdClang_Cpp_Compiler, Pthread_Pgi_Cpp_Compiler,
                      Pthread_AppleClang_Cpp_Compiler]

        elif backend == "cuda":
            clist =  [Cuda_Cpp_Compiler]

        elif backend == "hip":
            clist =  [Hip_Cpp_Compiler]

        elif backend == "openacc-c++":
            clist =  [OpenAcc_Gnu_Cpp_Compiler, OpenAcc_CrayClang_Cpp_Compiler,
                      OpenAcc_Pgi_Cpp_Compiler]

        elif backend == "fortran":
            clist =  [AmdFlang_Fortran_Compiler, Cray_Fortran_Compiler,
                        Pgi_Fortran_Compiler, IbmXl_Fortran_Compiler,
                        AppleGnu_Fortran_Compiler, Gnu_Fortran_Compiler]

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

