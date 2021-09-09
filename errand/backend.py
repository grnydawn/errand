"""Errand backend module


"""

import os, sys, abc, hashlib
import numpy as np
from numpy.ctypeslib import ndpointer, load_library
from ctypes import c_int, c_longlong, c_float, c_double, c_size_t

from errand.util import shellcmd, split_compile

_installed_backends = {}

varclass_template = """
class {vartype} {{
public:
    {dtype} * data;
    int * _attrs; // ndim, itemsize, size, shape, strides

    {funcprefix} {dtype}& operator() ({oparg}) {{
        int * s = &(_attrs[3+_attrs[0]]);
        return data[{offset}];
    }}
    {funcprefix} {dtype} operator() ({oparg}) const {{
        int * s = &(_attrs[3+_attrs[0]]);
        return data[{offset}];
    }}

    {funcprefix} int ndim() {{
        return _attrs[0];
    }}
    {funcprefix} int itemsize() {{
        return _attrs[1];
    }}
    {funcprefix} int size() {{
        return _attrs[2];
    }}
    {funcprefix} int shape(int dim) {{
        return _attrs[3+dim];
    }}
    {funcprefix} int stride(int dim) {{
        return _attrs[3+_attrs[0]+dim];
    }}
    {funcprefix} int unravel_index(int tid, int dim) {{
        int q, r=tid, s;
        for (int i = 0; i < dim + 1; i++) {{
            s = stride(i);
            q = r / s;
            r = r % s;
        }}
        return q;
    }}
}};
"""


class Backend(abc.ABC):
    """Errand Backend class

    * keep as transparent and passive as possible
"""
    code_template = """
{top}

{header}

{namespace}

{varclass}

{struct}

int isfinished = 0;

{vardef}

{varglobal}

{function}

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

    def __init__(self, workdir, compilers, targetsystem):

        self.workdir = workdir
        self.sharedlib = None
        self.nteams = None
        self.nmembers = None
        self.nassigns = None
        self.inargs = None
        self.outargs = None
        self.order = None
        self.compilers = compilers
        self.hostsystem = None
        self.targetsystem = targetsystem

    # TODO: need update to support multiple compilers and target systems
    def isavail(self):
        return (self.compilers is not None and self.compilers.isavail() and
                self.targetsystem is not None and self.targetsystem.isavail())

    def get_compiler(self):
        
        return self.compilers.select_one()

    def code_top(self):
        return ""

    def code_header(self):
        return ""

    def code_namespace(self):
        return ""

    def code_varclass(self):
        return ""

    def code_struct(self):
        return ""

    def code_vardef(self):
        return ""

    def code_varglobal(self):
        return ""

    def code_h2dcopyfunc(self):
        return ""

    def code_d2hcopyfunc(self):
        return ""

    @abc.abstractmethod
    def code_devfunc(self):
        pass

    def code_function(self):
        return ""

    def code_prerun(self):
        return ""

    @abc.abstractmethod
    def code_calldevmain(self):
        pass

    def code_postrun(self):
        return ""

    def code_tail(self):
        return ""

    def getname_argpair(self, arg):
        return (arg["data"].ndim, self.getname_ctype(arg))

    def get_ctype(self, arg):
       
        return self.dtypemap[arg["data"].dtype.name][1]

    def getname_ctype(self, arg):
       
        return self.dtypemap[arg["data"].dtype.name][0]
 
    def gencode(self, nteams, nmembers, nassigns, inargs, outargs, order):

        innames, outnames = order.get_argnames()

        if innames or outnames:
            assert len(innames) == len(inargs), "The number of input arguments mismatches."
            assert len(outnames) == len(outargs), "The number of input arguments mismatches."

            for arg, name in zip(inargs+outargs, innames+outnames):
                arg["curname"] = name

        self.nteams = nteams
        self.nmembers = nmembers
        self.nassigns = nassigns
        self.inargs = inargs
        self.outargs = outargs
        self.order = order

        # generate source code
        top = self.code_top()
        header = self.code_header()
        namespace = self.code_namespace()
        varclass = self.code_varclass()
        struct = self.code_struct()
        vardef = self.code_vardef()
        varglobal = self.code_varglobal()
        h2dcopyfunc = self.code_h2dcopyfunc()
        d2hcopyfunc = self.code_d2hcopyfunc()
        devfunc = self.code_devfunc()
        function = self.code_function()
        prerun = self.code_prerun()
        calldevmain = self.code_calldevmain()
        postrun = self.code_postrun()
        tail = self.code_tail()

        code = self.code_template.format(top=top, header=header,
            namespace=namespace, varclass=varclass, vardef=vardef,
            h2dcopyfunc=h2dcopyfunc, d2hcopyfunc=d2hcopyfunc,
            devfunc=devfunc, prerun=prerun, calldevmain=calldevmain,
            postrun=postrun, tail=tail, struct=struct, function=function,
            varglobal=varglobal)

        fname = hashlib.md5(code.encode("utf-8")).hexdigest()[:10]

        codepath = os.path.join(self.workdir, fname + "." + self.codeext)
        with open(codepath, "w") as f:
            f.write(code)

        # generate shared library
        # TODO : retry compilation for debug and performance optimization

        libpath = os.path.join(self.workdir, fname + "." + self.libext)

        compiler = self.get_compiler()
        if compiler is None:
            raise Exception("Compiler is not available.")

        cmd = "%s %s -o %s %s" % (compiler.path, compiler.get_option(), libpath,
                                  codepath)

        #import pdb; pdb.set_trace()
        out = shellcmd(cmd)

        if out.returncode  != 0:
            print(out.stderr.decode())
            sys.exit(out.returncode)

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


def select_backends(backend, compile, order, workdir):

    if len(_installed_backends) == 0:
        from errand.cuda_hip import CudaBackend, HipBackend
        from errand.pthread import PThreadBackend
        from errand.openacc import OpenAccCppBackend

        _installed_backends[CudaBackend.name] = CudaBackend
        _installed_backends[HipBackend.name] = HipBackend
        _installed_backends[PThreadBackend.name] = PThreadBackend
        _installed_backends[OpenAccCppBackend.name] = OpenAccCppBackend

    candidate = None

    if isinstance(backend, Backend):
        candidate = backend.__class__

    if candidate is None and isinstance(backend, str):
        if backend in _installed_backends:
            candidate = _installed_backends[backend]
        else:
            raise Exception("%s backend is not installed." % str(backend))

    selected = []
    targets = order.get_backends()
        
    for tname in targets:
        if tname in _installed_backends:
            tempeng = _installed_backends[tname]
            s = None

            if candidate is not None:
                if candidate is tempeng:
                    s = tempeng
            else:
                s = tempeng

            if s is not None:

                if compile is None:
                    compile = split_compile(order.get_section(tname).arg)

                b = s(workdir, compile)
                if b.isavail():
                    selected.append(b)

    if len(selected) == 0:
        if backend is None:
            raise Exception(("A compiler for any of these errand backends (%s)"
                    "is not found on this system.") %
                    ", ".join(_installed_backends.keys()))

        elif candidate is not None:

            if backend not in targets:
                raise Exception("Backend, '%s' is not specified in order." % str(backend))

            else:
                raise Exception("Backend, '%s' is not supported by the system." % str(backend))

        else:
            raise Exception("Unknown backend: %s" % str(backend))
    else:
        return selected