"""Errand Cuda and Hip engine module


"""

import os, sys, abc
import subprocess as subp

from numpy import double
from numpy.ctypeslib import load_library

from errand.engine import Engine
from errand.util import which

# key ndarray attributes
# shape, dtype, strides, itemsize, ndim, flags, size, nbytes
# flat, ctypes, reshape

# TODO: follow ndarray convention to copy data between CPU and GPU
# TODO: send data and array of attributes to an internal variable of generated struct
#       the attribute array will be interpreted within the struct to various info


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
        //int s = 3+_attrs[0];
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
}};
"""

host_vardef_template = """
{vartype} {varname} = {vartype}();
"""

dev_vardef_template = """
{vartype} {varname} = {vartype}();
"""

hip_h2dcopy_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    hipMalloc((void **)&{dvar}.data, {hvar}.size() * sizeof({dtype}));
    hipMalloc((void **)&{dvar}._attrs, attrsize * sizeof(int));

    hipMemcpyHtoD({dvar}.data, {hvar}.data, {hvar}.size() * sizeof({dtype}));
    hipMemcpyHtoD({dvar}._attrs, {hvar}._attrs, attrsize * sizeof(int));

    return 0;
}}
"""

hip_h2dmalloc_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    hipMalloc((void **)&{dvar}.data, {hvar}.size() * sizeof({dtype}));
    hipMalloc((void **)&{dvar}._attrs, attrsize * sizeof(int));

    //hipMemcpyHtoD({dvar}.data, {hvar}.data, {hvar}.size() * sizeof({dtype}));
    hipMemcpyHtoD({dvar}._attrs, {hvar}._attrs, attrsize * sizeof(int));

    return 0;
}}
"""

hip_d2hcopy_template = """
extern "C" int {name}(void * data) {{

    hipMemcpyDtoH(data, {dvar}.data, {hvar}.size() * sizeof({dtype}));

    return 0;
}}
"""

cuda_h2dcopy_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    cudaMalloc((void **)&({dvar}.data), {hvar}.size() * sizeof({dtype}));
    cudaMalloc((void **)&({dvar}._attrs), attrsize * sizeof(int));

    cudaMemcpy({dvar}.data, {hvar}.data, {hvar}.size() * sizeof({dtype}), cudaMemcpyHostToDevice);
    cudaMemcpy({dvar}._attrs, {hvar}._attrs, attrsize * sizeof(int), cudaMemcpyHostToDevice);

    return 0;
}}
"""

cuda_h2dmalloc_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    cudaMalloc((void **)&({dvar}.data), {hvar}.size() * sizeof({dtype}));
    cudaMalloc((void **)&({dvar}._attrs), attrsize * sizeof(int));

    cudaMemcpy({dvar}._attrs, {hvar}._attrs, attrsize * sizeof(int), cudaMemcpyHostToDevice);

    return 0;
}}
"""

cuda_d2hcopy_template = """
extern "C" int {name}(void * data) {{

    cudaMemcpy(data, {dvar}.data, {hvar}.size() * sizeof({dtype}), cudaMemcpyDeviceToHost);

    return 0;
}}
"""

devfunc_template = """
__global__ void _kernel({args}){{
    {body}
}}
"""

calldevmain_template = """
    _kernel<<<{ngrids}, {nthreads}>>>({args});
"""

class CudaHipEngine(Engine):

    def __init__(self, workdir):

        super(CudaHipEngine, self).__init__(workdir)

    def getname_h2dcopy(self, arg):

        name = self.argmap[id(arg)]
        return "h2dcopy_%s" % name
      
    def getname_h2dmalloc(self, arg):

        name = self.argmap[id(arg)]
        return "h2dmalloc_%s" % name

    def getname_d2hcopy(self, arg):

        name = self.argmap[id(arg)]
        return "d2hcopy_%s" % name

    def getname_vartype(self, arg, devhost):

        aname, ndim, dname = self.getname_argtriple(arg)
        return "%s_%s_dim%s" % (devhost, dname, ndim)

    def getname_var(self, arg, devhost):

        return devhost + "_" + self.argmap[id(arg)]

    def len_argattrs(self, arg):

        return 3 + len(arg.shape)*2

    def get_argattrs(self, arg):

        return ((arg.ndim, arg.itemsize, arg.size) + arg.shape +
                tuple([int(s//arg.itemsize) for s in arg.strides]))

    def code_varclass(self):

        dvs = {}

        for arg, attr in self.inargs+self.outargs:

            aname, ndim, dname = self.getname_argtriple(arg)

            if dname in dvs:
                dvsd = dvs[dname]

            else:
                dvsd = {}
                dvs[dname] = dvsd
                
            if ndim not in dvsd:
                oparg = ", ".join(["int dim%d"%d for d in range(arg.ndim)])
                offset = "+".join(["s[%d]*dim%d"%(d,d) for d in range(arg.ndim)])
                attrsize = self.len_argattrs(arg)

                hvartype = self.getname_vartype(arg, "host")
                out = varclass_template.format(vartype=hvartype, oparg=oparg,
                        offset=offset, funcprefix="", dtype=dname,
                        attrsize=attrsize)

                dvartype = self.getname_vartype(arg, "dev")
                out += varclass_template.format(vartype=dvartype, oparg=oparg,
                        offset=offset, funcprefix="__device__", dtype=dname,
                        attrsize = attrsize)

                dvsd[ndim] = out

        return "\n".join([y for x in dvs.values() for y in x.values()])

    def code_vardef(self):

        out = ""

        for arg, attr in self.inargs+self.outargs:

            aname, ndim, dname = self.getname_argtriple(arg)

            out += host_vardef_template.format(vartype=self.getname_vartype(arg,
                    "host"), varname=self.getname_var(arg, "host"))

            out += dev_vardef_template.format(vartype=self.getname_vartype(arg,
                    "dev"), varname=self.getname_var(arg, "dev"))

        return out

    def code_devfunc(self):

        args = []
        body = "\n".join(self.order.get_section(self.name)[2])

        for arg, attr in self.inargs+self.outargs:

            aname, ndim, dname = self.getname_argtriple(arg)

            args.append("dev_%s_dim%s %s" % (dname, ndim, aname))

        return devfunc_template.format(args=", ".join(args), body=body)

    def code_h2dcopyfunc(self):

        out = ""

        for arg, attr in self.inargs:

            aname, ndim, dname = self.getname_argtriple(arg)
            fname = self.getname_h2dcopy(arg)

            template = self.get_template("h2dcopy")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            vartype = self.getname_vartype(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname, dtype=dname, vartype=vartype)

        for arg, attr in self.outargs:

            aname, ndim, dname = self.getname_argtriple(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            vartype = self.getname_vartype(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname, dtype=dname, vartype=vartype)

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for aname, (arg, attr) in zip(self.outnames, self.outargs):

            aname, ndim, dname = self.getname_argtriple(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname, dtype=dname)

        return out

    def code_calldevmain(self):

        args = []

        for aname, (arg, attr) in zip(self.innames+self.outnames,
            self.inargs+self.outargs):

            args.append(self.getname_var(arg, "dev"))

        return calldevmain_template.format(ngrids=str(self.nteams),
                nthreads=str(self.nmembers), args=", ".join(args))

    def compiler_path(self):
        return self.compiler


class CudaEngine(CudaHipEngine):

    name = "cuda"
    codeext = "cu"
    libext = "so"

    def __init__(self, workdir):

        super(CudaEngine, self).__init__(workdir)

        compiler = which("nvcc")
        if compiler is None or not self.isavail():
            raise Exception("nvcc is not found")

        self.compiler = os.path.realpath(compiler)
        self.option = ""

    def compiler_option(self):
        return self.option + "--compiler-options '-fPIC' --shared"

    @classmethod
    def isavail(cls):

        compiler = which("nvcc")
        if compiler is None or not os.path.isfile(compiler):
            return False

        rootdir = os.path.join(os.path.dirname(compiler), "..")

        incdir = os.path.join(rootdir, "include")
        if not os.path.isdir(incdir):
            return False

        libdir = os.path.join(rootdir, "lib64")
        if not os.path.isdir(libdir):
            libdir = os.path.join(rootdir, "lib")

            if not os.path.isdir(libdir):
                return False

        return True

    def code_header(self):

        return  "#include <stdexcept>"

    def get_template(self, name):

        if name == "h2dcopy":
            return cuda_h2dcopy_template

        elif name == "h2dmalloc":
            return cuda_h2dmalloc_template

        elif name == "d2hcopy":
            return cuda_d2hcopy_template

    def code_header(self):

        out = """#include "stdio.h"
"""
        return out


class HipEngine(CudaHipEngine):

    name = "hip"
    codeext = "hip.cpp"
    libext = "so"

    def __init__(self, workdir):

        super(HipEngine, self).__init__(workdir)

        compiler = which("hipcc")
        if compiler is None or not self.isavail():
            raise Exception("hipcc is not found")

        self.compiler = os.path.realpath(compiler)
        self.option = ""

    def compiler_option(self):
        return self.option + " -fPIC --shared -O0"

    @classmethod
    def isavail(cls):

        compiler = which("hipcc")
        if compiler is None or not os.path.isfile(compiler):
            return False

        rootdir = os.path.join(os.path.dirname(compiler), "..")

        incdir = os.path.join(rootdir, "include")
        if not os.path.isdir(incdir):
            return False

        libdir = os.path.join(rootdir, "lib64")
        if not os.path.isdir(libdir):
            libdir = os.path.join(rootdir, "lib")

            if not os.path.isdir(libdir):
                return False

        return True

    def code_header(self):

        out = """#include <stdexcept>
#include <hip/hip_runtime.h>"""

        return out

    def get_template(self, name):

        if name == "h2dcopy":
            return hip_h2dcopy_template

        elif name == "h2dmalloc":
            return hip_h2dmalloc_template

        elif name == "d2hcopy":
            return hip_d2hcopy_template
