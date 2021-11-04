"""Errand Fortran backend module


"""

import os
import numpy

from errand.backend import FortranBackendBase, fortran_varclass_template
from errand.compiler import Compilers
from errand.system import select_system
from errand.util import which


struct_template = """
"""

host_vardef_template = """
"""

varglobal_template = """
"""

pthrd_h2dcopy_template = """
INTEGER FUNCTION {name} (data, attrs, attrsize)
    {dtype}, DIMENSION(:), INTENT(IN) :: data
    INTEGER, DIMENSION(:), INTENT(IN) :: attrs
    INTEGER, INTENT(IN) :: attrsize

    !PRINT *, data
    !{hvar}@data = data
    !ALLOCATE({hvar}@attrs(attrsize))
    !{hvar}@attrs = attrs

    {name} = 0

END FUNCTION
"""

pthrd_h2dmalloc_template = """
INTEGER FUNCTION {name} (data, attrs, attrsize)
    {dtype}, DIMENSION(:), INTENT(IN) :: data
    INTEGER, DIMENSION(:), INTENT(IN) :: attrs
    INTEGER, INTENT(IN) :: attrsize

    !PRINT *, data
    !{hvar}@data = data
    !ALLOCATE({hvar}@attrs(attrsize))
    !{hvar}@attrs = attrs

    {name} = 0

END FUNCTION

"""

pthrd_d2hcopy_template = """
INTEGER FUNCTION {name} (data)
    {dtype}, DIMENSION(:), INTENT(IN) :: data

    !PRINT *, data
    !{hvar}@data = data

    {name} = 0

END FUNCTION

"""

devfunc_template = """
"""

function_template = """
"""

calldevmain_template = """
"""

class FortranBackend(FortranBackendBase):

    name = "fortran"
    codeext = "f90"
    libext = "so"

    def __init__(self, workdir, compile):

        compilers = Compilers(self.name, compile)
        targetsystem = select_system("cpu")

        super(FortranBackend, self).__init__(workdir, compilers,
            targetsystem)

    #def compiler_option(self):
    #    return self.option + "--compiler-options '-fPIC' --shared"

    def code_header(self):

        return  """
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
"""

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
                dvsd[ndim] = fortran_varclass_template.format(vartype=hvartype, oparg=oparg,
                        offset=offset, funcprefix="", dtype=dname,
                        attrsize=attrsize)

        return "\n".join([y for x in dvs.values() for y in x.values()])

    def code_struct(self):

        out = []

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)
            out.append("%s * %s;" % (self.getname_vartype(arg, "host"),
                        self.getname_var(arg, "host")))

        #out.append("int tid;")

        return struct_template.format(args="\n".join(out))

    def code_varglobal(self):

        out = []

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            varname = self.getname_var(arg, "host")

            out.append(".{name} = &{name}".format(name=varname))

        return varglobal_template.format(varassign=",\n".join(out))

    def code_vardef(self):

        out = ""

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            out += host_vardef_template.format(vartype=self.getname_vartype(arg,
                    "host"), varname=self.getname_var(arg, "host"))

        return out

    def code_function(self):

        nthreads = numpy.prod(self.nteams) * numpy.prod(self.nmembers)
        return function_template.format(nthreads=str(nthreads))

    def code_devfunc(self):

        argdef = []
        argassign = []

        body = str(self.order.get_section(self.name))

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            #argdef.append("host_%s_dim%s %s = host_%s_dim%s();" % (dname, ndim, arg["curname"], dname, ndim))
            argdef.append("host_%s_dim%s %s;" % (dname, ndim, arg["curname"]))
            argassign.append("%s = *(args->data->host_%s);" % (arg["curname"], arg["curname"]))

        argassign.append("int ERRAND_GOFER_ID = 0;")

        return devfunc_template.format(argdef="\n".join(argdef), body=body,
                    argassign="\n".join(argassign))

    def code_h2dcopyfunc(self):

        out = ""

        for arg in self.inargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dcopy(arg)

            template = self.get_template("h2dcopy")
            hvar = self.getname_var(arg, "host")
            out += template.format(hvar=hvar, name=fname, dtype=dname)

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")
            hvar = self.getname_var(arg, "host")
            out += template.format(hvar=hvar, name=fname, dtype=dname)

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")
            hvar = self.getname_var(arg, "host")
            out += template.format(hvar=hvar, name=fname, dtype=dname)

        return out

 
    def code_calldevmain(self):
#
#        argassign = []
#
#        for arg in self.inargs+self.outargs:
#
#            args.append(self.getname_var(arg, "host"))
#
        # testing
        #args.append("1")

        nthreads = numpy.prod(self.nteams) * numpy.prod(self.nmembers)
        return calldevmain_template.format(nthreads=str(nthreads))

    def get_template(self, name):

        if name == "h2dcopy":
            return pthrd_h2dcopy_template

        elif name == "h2dmalloc":
            return pthrd_h2dmalloc_template

        elif name == "d2hcopy":
            return pthrd_d2hcopy_template
