"""Errand Cuda and Hip engine module


"""

import os

from errand.engine import Engine, varclass_template
from errand.compiler import Compilers
from errand.system import select_system
from errand.util import which

# key ndarray attributes
# shape, dtype, strides, itemsize, ndim, flags, size, nbytes
# flat, ctypes, reshape

# TODO: follow ndarray convention to copy data between CPU and GPU
# TODO: send data and array of attributes to an internal variable of generated struct
#       the attribute array will be interpreted within the struct to various info


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

    def __init__(self, workdir, compilers, targetsystem):

        super(CudaHipEngine, self).__init__(workdir, compilers,
                targetsystem)

    def getname_h2dcopy(self, arg):

        return "h2dcopy_%s" % arg["curname"]
      
    def getname_h2dmalloc(self, arg):

        return "h2dmalloc_%s" % arg["curname"]

    def getname_d2hcopy(self, arg):

        return "d2hcopy_%s" % arg["curname"]

    def getname_vartype(self, arg, devhost):

        ndim, dname = self.getname_argpair(arg)
        return "%s_%s_dim%s" % (devhost, dname, ndim)

    def getname_var(self, arg, devhost):

        return devhost + "_" + arg["curname"]

    def len_numpyattrs(self, arg):

        return 3 + len(arg["data"].shape)*2

    def get_numpyattrs(self, arg):
        data = arg["data"]

        return ((data.ndim, data.itemsize, data.size) + data.shape +
                tuple([int(s//data.itemsize) for s in data.strides]))

    def code_varclass(self):

        dvs = {}

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            if dname in dvs:
                dvsd = dvs[dname]

            else:
                dvsd = {}
                dvs[dname] = dvsd
                
            if ndim not in dvsd:
                oparg = ", ".join(["int dim%d"%d for d in
                                    range(arg["data"].ndim)])
                offset = "+".join(["s[%d]*dim%d"%(d,d) for d in
                                    range(arg["data"].ndim)])
                attrsize = self.len_numpyattrs(arg)

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

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            out += host_vardef_template.format(vartype=self.getname_vartype(arg,
                    "host"), varname=self.getname_var(arg, "host"))

            out += dev_vardef_template.format(vartype=self.getname_vartype(arg,
                    "dev"), varname=self.getname_var(arg, "dev"))

        return out

    def code_devfunc(self):

        args = []
        body = str(self.order.get_section(self.name))

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            args.append("dev_%s_dim%s %s" % (dname, ndim, arg["curname"]))

        return devfunc_template.format(args=", ".join(args), body=body)

    def code_h2dcopyfunc(self):

        out = ""

        for arg in self.inargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dcopy(arg)

            template = self.get_template("h2dcopy")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            vartype = self.getname_vartype(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname,
                                    dtype=dname, vartype=vartype)

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            vartype = self.getname_vartype(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname,
                                    dtype=dname, vartype=vartype)

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")
            hvar = self.getname_var(arg, "host")
            dvar = self.getname_var(arg, "dev")
            out += template.format(hvar=hvar, dvar=dvar, name=fname, dtype=dname)

        return out

    def code_calldevmain(self):

        args = []

        for arg in self.inargs+self.outargs:

            args.append(self.getname_var(arg, "dev"))

        return calldevmain_template.format(ngrids=str(self.nteams),
                nthreads=str(self.nmembers), args=", ".join(args))

    def code_postrun(self):
        # NOTE: mark finished here; d2h copy will block until gpu run finish.
        return "    isfinished = 1;";


class CudaEngine(CudaHipEngine):

    name = "cuda"
    codeext = "cu"
    libext = "so"

    def __init__(self, workdir):

        compilers = Compilers(self.name)
        targetsystem = select_system("nvidia-gpu")

        super(CudaEngine, self).__init__(workdir, compilers,
            targetsystem)

    def code_header(self):

        return  """
#include <stdexcept>
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
"""

    def get_template(self, name):

        if name == "h2dcopy":
            return cuda_h2dcopy_template

        elif name == "h2dmalloc":
            return cuda_h2dmalloc_template

        elif name == "d2hcopy":
            return cuda_d2hcopy_template


class HipEngine(CudaHipEngine):

    name = "hip"
    codeext = "hip.cpp"
    libext = "so"

    def __init__(self, workdir):

        compilers = Compilers(self.name)
        targetsystem = select_system("amd-gpu")

        super(HipEngine, self).__init__(workdir, compilers,
            targetsystem)

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
