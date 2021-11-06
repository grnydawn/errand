"""Errand Fortran backend module


"""

import os
import numpy

from errand.backend import (FortranBackendBase, fortran_attrtype_template,
                            fortran_attrproc_template)
from errand.compiler import Compilers
from errand.system import select_system
from errand.util import which


struct_template = """
"""


pthrd_h2dcopy_template = """
INTEGER (C_INT) FUNCTION {name} (data, attrs, attrsize_) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING 
    USE global, ONLY : {varname}, {attrname}
    IMPLICIT NONE 
    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data
    INTEGER (C_INT), DIMENSION(*), INTENT(IN) :: attrs
    INTEGER (C_INT), INTENT(IN) :: attrsize_
    INTEGER i, j

    {varname} => data
    ALLOCATE({attrname})
    ALLOCATE({attrname}%attrs({attrsize}))
    {attrname}%attrs(:) = attrs(1:{attrsize})

!    DO i=1,{attrname}%shape(1)
!        DO j=1,{attrname}%shape(2)
!            print *, {varname}(i, j)
!        END DO
!    END DO

    {name} = 0

END FUNCTION
"""

pthrd_h2dmalloc_template = """
INTEGER (C_INT) FUNCTION {name} (data, attrs, attrsize_) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING 
    USE global, ONLY : {varname}, {attrname}
    IMPLICIT NONE 
    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data
    INTEGER (C_INT), DIMENSION(*), INTENT(IN) :: attrs
    INTEGER (C_INT), INTENT(IN) :: attrsize_
    INTEGER i, j

    {varname} => data
    ALLOCATE({attrname})
    ALLOCATE({attrname}%attrs({attrsize}))
    {attrname}%attrs(:) = attrs(1:{attrsize})

    !print *, {attrname}%size()

    {name} = 0

END FUNCTION

"""

pthrd_d2hcopy_template = """
INTEGER (C_INT) FUNCTION {name} (data) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING 
    USE global, ONLY : {varname}, {attrname}
    IMPLICIT NONE 
    {dtype}, DIMENSION({bound}), INTENT(OUT) :: data

    data = {varname}

    {name} = 0

END FUNCTION

"""

devfunc_template = """
"""

function_template = """
"""

calldevmain_template = """
{body}
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

    def code_attrtype(self):

        return fortran_attrtype_template

    def code_attrproc(self):

        return fortran_attrproc_template

    def code_varattr(self):

        data = []

        for arg in self.inargs+self.outargs:

            data.append(arg["curname"])
            data.append(arg["curname"]+"_")

        return ",".join(data)

    def code_struct(self):

        out = []

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)
            out.append("%s * %s;" % (self.getname_vartype(arg, "host"),
                        self.getname_var(arg, "host")))

        #out.append("int tid;")

        return struct_template.format(args="\n".join(out))

    def code_attrdef(self):

        out = ""

        for arg in self.inargs+self.outargs:

            out += "CLASS(attrtype), ALLOCATABLE :: %s\n" % (arg["curname"]+"_")

        return out

#        out = []
#
#        for arg in self.inargs+self.outargs:
#
#            ndim, dname = self.getname_argpair(arg)
#
#            varname = self.getname_var(arg, "host")
#
#            out.append(".{name} = &{name}".format(name=varname))
#
#        return attrdef_template.format(varassign=",\n".join(out))

    def code_vardef(self):

        out = ""

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)
            shape = ",".join((":",)*ndim)
            out += "%s, DIMENSION(%s), POINTER :: %s\n" % (dname, shape, arg["curname"])

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

            bound = []
            for s in arg["data"].shape:
                bound.append("%d" % s)

            attrsize = self.len_numpyattrs(arg)

            out += template.format(name=fname, dtype=dname,
                    varname=arg["curname"], attrname=arg["curname"]+"_",
                    bound=",".join(bound), attrsize=str(attrsize))

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")

            bound = []
            for s in arg["data"].shape:
                bound.append("%d" % s)

            attrsize = self.len_numpyattrs(arg)

            out += template.format(name=fname, dtype=dname,
                    varname=arg["curname"], attrname=arg["curname"]+"_",
                    bound=",".join(bound), attrsize=str(attrsize))

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")

            bound = []
            for s in arg["data"].shape:
                bound.append("%d" % s)

            out += template.format(name=fname, dtype=dname,
                    varname=arg["curname"], attrname=arg["curname"]+"_",
                    bound=",".join(bound))

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

        #nthreads = numpy.prod(self.nteams) * numpy.prod(self.nmembers)
        #return calldevmain_template.format(nthreads=str(nthreads))

        body = str(self.order.get_section(self.name))

        return calldevmain_template.format(body=body)

    def get_template(self, name):

        if name == "h2dcopy":
            return pthrd_h2dcopy_template

        elif name == "h2dmalloc":
            return pthrd_h2dmalloc_template

        elif name == "d2hcopy":
            return pthrd_d2hcopy_template
