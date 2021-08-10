"""Errand Cuda and Hip engine module


"""

import os, sys, abc
import subprocess as subp

from numpy import double
from numpy.ctypeslib import load_library

from errand.engine import Engine
from errand.util import which

# key ndarray attributes
# shape, dtype, strides, itemsize, ndims, flags, size, nbytes
# flat, ctypes, reshape

# TODO: follow ndarray convention to copy data between CPU and GPU
# TODO: send data and array of attributes to an internal variable of generated struct
#       the attribute array will be interpreted within the struct to various info


varclass_template = """
class {devhost}_{dtype}_dim{ndims} {{
public:
    {dtype} * data;
    int * _attrs; // ndims, itemsize, size, strides

    {funcprefix} int ndims() {{
        return _attrs[0];
    }}
    {funcprefix} int itemsize() {{
        return _attrs[1];
    }}
    {funcprefix} int size() {{
        return _attrs[2];
    }}
}};
"""

host_vardef_template = """
host_{dtype}_dim{ndims} h_{arg};
"""

dev_vardef_template = """
__device__ dev_{dtype}_dim{ndims} d_{arg};
"""

#extern "C" int h2dcopy_{arg}(void * data, int size) {{
hip_h2dcopy_template = """
extern "C" int {name}(void * data, void * _attrs) {{

    h_{arg}.data = ({dtype} *) data;
    h_{arg}._attrs = (int *) _attrs;
    int attrsize = 3 + h_{arg}.ndims(); // should match with get_argattrs

    hipMalloc((void **)&d_{arg}.data, h_{arg}.size() * sizeof({dtype}));
    hipMalloc((void **)&d_{arg}._attrs, attrsize * sizeof(int));

    hipMemcpyHtoD(d_{arg}.data, h_{arg}.data, h_{arg}.size() * sizeof({dtype}));
    hipMemcpyHtoD(d_{arg}._attrs, h_{arg}._attrs, attrsize * sizeof(int));

    return 0;
}}
"""

hip_h2dmalloc_template = """
extern "C" int {name}(void * data, void * _attrs) {{

    h_{arg}.data = ({dtype} *) data;
    h_{arg}._attrs = (int *) _attrs;
    int attrsize = 3 + h_{arg}.ndims(); // should match with get_argattrs

    hipMalloc((void **)&d_{arg}.data, h_{arg}.size() * sizeof({dtype}));
    hipMalloc((void **)&d_{arg}._attrs, attrsize * sizeof(int));

    hipMemcpyHtoD(d_{arg}._attrs, h_{arg}._attrs, attrsize * sizeof(int));

    return 0;
}}
"""

hip_d2hcopy_template = """
extern "C" int {name}(void * data, int size) {{

    hipMemcpyDtoH(h_{arg}.data, d_{arg}.data, size * sizeof({dtype}));

    data = (void *) h_{arg}.data;

    return 0;
}}
"""

cuda_h2dcopy_template = """
extern "C" int {name}(void * data, int size) {{
    h_{arg}.data = ({dtype} *) data;
    cudaMalloc((void **)&d_{arg}.data, size * sizeof({dtype}));
    cudaMalloc((void **)&d_{arg}._size, sizeof(int));
    cudaMemcpy(d_{arg}.data, h_{arg}.data, size * sizeof({dtype}), cudaMemcpyHostToDevice);
    cudaMemcpy(d_{arg}._size, &size, sizeof(int), cudaMemcpyHostToDevice);
    return 0;
}}
"""

cuda_h2dmalloc_template = """
extern "C" int {name}(void * data, int size) {{
    h_{arg}.data = ({dtype} *) data;
    cudaMalloc((void **)&d_{arg}.data, size * sizeof({dtype}));
    cudaMalloc((void **)&d_{arg}._size, sizeof(int));
    cudaMemcpy(d_{arg}._size, &size, sizeof(int), cudaMemcpyHostToDevice);
    return 0;
}}
"""

cuda_d2hcopy_template = """
extern "C" int {name}(void * data, int size) {{
    cudaMemcpy(h_{arg}.data, d_{arg}.data, size * sizeof({dtype}), cudaMemcpyDeviceToHost);
    data = (void *) h_{arg}.data;
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

    def get_argattrs(self, arg):

        return (arg.ndim, arg.itemsize, arg.size) + arg.strides

    def code_varclass(self):

        dvs = {}

        for arg, attr in self.inargs+self.outargs:

            aname, ndims, dname = self.getname_argtriple(arg)

            if dname in dvs:
                dvsd = dvs[dname]

            else:
                dvsd = {}
                dvs[dname] = dvsd
                
            if ndims not in dvsd:
                out = varclass_template.format(devhost="host", funcprefix="",
                            dtype=dname, ndims=ndims)
                out += varclass_template.format(devhost="dev",
                            funcprefix="__device__", dtype=dname, ndims=ndims)
                dvsd[ndims] = out

        return "\n".join([y for x in dvs.values() for y in x.values()])

    def code_vardef(self):

        out = ""

        for arg, attr in self.inargs+self.outargs:

            aname, ndims, dname = self.getname_argtriple(arg)

            out += host_vardef_template.format(arg=aname, ndims=ndims, dtype=dname)
            out += dev_vardef_template.format(arg=aname, ndims=ndims, dtype=dname)

        return out

    def code_devfunc(self):

        args = []
        body = "\n".join(self.order.get_section(self.name)[2])

        for arg, attr in self.inargs+self.outargs:

            aname, ndims, dname = self.getname_argtriple(arg)

            args.append("dev_%s_dim%s %s" % (dname, ndims, aname))

        return devfunc_template.format(args=", ".join(args), body=body)

    def code_h2dcopyfunc(self):

        out = ""

        for arg, attr in self.inargs:

            aname, ndims, dname = self.getname_argtriple(arg)
            fname = self.getname_h2dcopy(arg)

            template = self.get_template("h2dcopy")
            out += template.format(name=fname, arg=aname, dtype=dname)

        for arg, attr in self.outargs:

            aname, ndims, dname = self.getname_argtriple(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")
            out += template.format(name=fname, arg=aname, dtype=dname)

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for aname, (arg, attr) in zip(self.outnames, self.outargs):

            aname, ndims, dname = self.getname_argtriple(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")
            out += template.format(name=fname, arg=aname, dtype=dname)

        return out

    def code_calldevmain(self):

        args = []

        for aname, (arg, attr) in zip(self.innames+self.outnames,
            self.inargs+self.outargs):

            args.append("d_"+aname)

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

        super(HipEngine, self).__init__(workdir)

        compiler = which("hipcc")
        if compiler is None or not self.isavail():
            raise Exception("hipcc is not found")

        self.compiler = os.path.realpath(compiler)
        self.option = ""

    def compiler_option(self):
        return self.option + " -fPIC --shared"

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

        return "#include <hip/hip_runtime.h>"

    def get_template(self, name):

        if name == "h2dcopy":
            return hip_h2dcopy_template

        elif name == "h2dmalloc":
            return hip_h2dmalloc_template

        elif name == "d2hcopy":
            return hip_d2hcopy_template
