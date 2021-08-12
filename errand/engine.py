"""Errand engine module


"""

import os, sys, abc, hashlib
import subprocess as subp
import numpy as np
from numpy.ctypeslib import ndpointer, load_library
from ctypes import c_int, c_longlong, c_float, c_double, c_size_t


_installed_engines = {}

class Engine(abc.ABC):
    """Errand Engine class

    * keep as transparent and passive as possible
"""
    code_template = """
{top}

{header}

{namespace}

{varclass}

int isfinished = 0;

{vardef}

{h2dcopyfunc}

{d2hcopyfunc}

{devfunc}

extern "C" int isalive() {{

    return isfinished;
}}

extern "C" int run() {{

    {prerun}

    {calldevmain} 

    {postrun}

    isfinished = 1;

    return 0;
}}
{tail}
"""

    dtypemap = {
        "int32": ["int", c_int],
        "int64": ["long", c_longlong],
        "float32": ["float", c_float],
        "float64": ["double", c_double]
    }

    def __init__(self, workdir):

        self.workdir = workdir
        self.sharedlib = None
        self.nteams = None
        self.nmembers = None
        self.inargs = None
        self.outargs = None
        self.order = None

    @classmethod
    @abc.abstractmethod
    def isavail(cls):
        pass

    def code_top(self):
        return ""

    def code_header(self):
        return ""

    def code_namespace(self):
        return ""

    def code_varclass(self):
        return ""

    def code_vardef(self):
        return ""

    def code_h2dcopyfunc(self):
        return ""

    def code_d2hcopyfunc(self):
        return ""

    @abc.abstractmethod
    def code_devfunc(self):
        pass

    def code_prerun(self):
        return ""

    @abc.abstractmethod
    def code_calldevmain(self):
        pass

    def code_postrun(self):
        return ""

    def code_tail(self):
        return ""

    @abc.abstractmethod
    def compiler_path(self):
        pass

    def getname_argpair(self, arg):
        return (arg["data"].ndim, self.getname_ctype(arg))

    @abc.abstractmethod
    def compiler_option(self):
        pass

    def get_ctype(self, arg):
       
        return self.dtypemap[arg["data"].dtype.name][1]

    def getname_ctype(self, arg):
       
        return self.dtypemap[arg["data"].dtype.name][0]
 
    def gencode(self, nteams, nmembers, inargs, outargs, order):

        innames, outnames = order.get_argnames()

        if innames or outnames:
            assert len(innames) == len(inargs), "The number of input arguments mismatches."
            assert len(outnames) == len(outargs), "The number of input arguments mismatches."

            for arg, name in zip(inargs+outargs, innames+outnames):
                arg["curname"] = name

        self.nteams = nteams
        self.nmembers = nmembers
        self.inargs = inargs
        self.outargs = outargs
        self.order = order

        # generate source code
        top = self.code_top()
        header = self.code_header()
        namespace = self.code_namespace()
        varclass = self.code_varclass()
        vardef = self.code_vardef()
        h2dcopyfunc = self.code_h2dcopyfunc()
        d2hcopyfunc = self.code_d2hcopyfunc()
        devfunc = self.code_devfunc()
        prerun = self.code_prerun()
        calldevmain = self.code_calldevmain()
        postrun = self.code_postrun()
        tail = self.code_tail()

        code = self.code_template.format(top=top, header=header,
            namespace=namespace, varclass=varclass, vardef=vardef,
            h2dcopyfunc=h2dcopyfunc, d2hcopyfunc=d2hcopyfunc,
            devfunc=devfunc, prerun=prerun, calldevmain=calldevmain,
            postrun=postrun, tail=tail)

        fname = hashlib.md5(code.encode("utf-8")).hexdigest()[:10]

        codepath = os.path.join(self.workdir, fname + "." + self.codeext)
        with open(codepath, "w") as f:
            f.write(code)

        # generate shared library
        # TODO : automated compiler option selection
        # TODO : retry compilation for debug and performance optimization

        libpath = os.path.join(self.workdir, fname + "." + self.libext)
        cmd = "%s %s -o %s %s" % (self.compiler_path(), self.compiler_option(),
                libpath, codepath)

        out = subp.run(cmd, shell=True, stdout=subp.PIPE, stderr=subp.PIPE, check=False)

        #import pdb; pdb.set_trace()
        if out.returncode  != 0:
            print(out.stderr)
            sys.exit(out.returncode)

        head, tail = os.path.split(libpath)
        base, ext = os.path.splitext(tail)

        # load the library
        self.sharedlib = load_library(base, head)

        # create a thread if required

        #return the library 
        return self.sharedlib
        
    @abc.abstractmethod
    def getname_h2dcopy(self, arg):
        pass
      
    @abc.abstractmethod
    def getname_h2dmalloc(self, arg):
        pass

    @abc.abstractmethod
    def getname_d2hcopy(self, arg):
        pass

    @abc.abstractmethod
    def get_numpyattrs(self, arg):
        pass

    def h2dcopy(self, inargs, outargs):

        # shape, dtype, strides, itemsize, ndims, flags, size, nbytes flat, ctypes, reshape

        for arg in inargs:

            attrs = self.get_numpyattrs(arg)
            cattrs = c_int*len(attrs)

            h2dcopy = getattr(self.sharedlib, self.getname_h2dcopy(arg))
            h2dcopy.restype = c_int
            h2dcopy.argtypes = [ndpointer(self.get_ctype(arg)), cattrs, c_int] 
            res = h2dcopy(arg["data"], cattrs(*attrs), len(attrs))

        for arg in outargs:

            attrs = self.get_numpyattrs(arg)
            cattrs = c_int*len(attrs)

            h2dmalloc = getattr(self.sharedlib, self.getname_h2dmalloc(arg))
            h2dmalloc.restype = c_int
            h2dmalloc.argtypes = [ndpointer(self.get_ctype(arg)), cattrs, c_int]
            res = h2dmalloc(arg["data"], cattrs(*attrs), len(attrs))

    def _copy2orgdata(self, arg):

        def _copyto(dst, src):

            if src.ndim == 1:
                for i, e in enumerate(src):
                    dst[i] = e
            else:
                for i in range(src.shape[0]):
                    _copyto(dst[i], src[i])

        if arg["data"].ndim == 0:
            raise Exception("Zero-dimension copy is not allowed.")

        else:
            _copyto(arg["orgdata"], arg["data"])

    def d2hcopy(self, outargs):

        for arg in outargs:

            d2hcopy = getattr(self.sharedlib, self.getname_d2hcopy(arg))
            d2hcopy.restype = c_int
            d2hcopy.argtypes = [ndpointer(self.get_ctype(arg))]

            res = d2hcopy(arg["data"])

            if type(arg["data"]) != type(arg["orgdata"]):
                self._copy2orgdata(arg)


def select_engine(engine, order):

    if len(_installed_engines) == 0:
        from errand.cuda_hip import CudaEngine, HipEngine

        _installed_engines[CudaEngine.name] = CudaEngine
        _installed_engines[HipEngine.name] = HipEngine

    if isinstance(engine, Engine):
        return engine.__class__

    if isinstance(engine, str):
        if engine in _installed_engines:
            return _installed_engines[engine]
    else:
        for tname in order.get_targetnames():
            if tname in _installed_engines and _installed_engines[tname].isavail():
                return _installed_engines[tname]

    if engine is None:
        raise Exception("A compiler for any of these errand engines (%s) is not found on this system."
                % ", ".join(_installed_engines.keys()))

    else:
        raise Exception("%s engine is not available." % str(engine))
