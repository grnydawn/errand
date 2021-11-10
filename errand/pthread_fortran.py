"""Errand PThread Fortran backend module


"""

import os
import numpy

from errand.backend import (FortranBackendBase, fortran_attrtype_template,
                            fortran_attrproc_template)
from errand.compiler import Compilers
from errand.system import select_system
from errand.fortscan import get_firstexec
from errand.util import which


typedef_template = """
    integer, parameter :: PTHREAD_SIZE       = 2    ! 8 Bytes.

    type, bind(c), public :: c_pthread_t
        private
        integer(kind=c_int) :: hidden(PTHREAD_SIZE)
    end type c_pthread_t

    interface
        ! int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)
        function c_pthread_create(thread, attr, start_routine, arg) bind(c, name='pthread_create')
            import :: c_int, c_ptr, c_funptr, c_pthread_t
            type(c_pthread_t), intent(inout)     :: thread
            type(c_ptr),       intent(in), value :: attr
            type(c_funptr),    intent(in), value :: start_routine
            type(c_ptr),       intent(in), value :: arg
            integer(kind=c_int)                  :: c_pthread_create
        end function c_pthread_create

        ! int pthread_join(pthread_t thread, void **value_ptr)
        function c_pthread_join(thread, value_ptr) bind(c, name='pthread_join')
            import :: c_int, c_ptr, c_pthread_t
            type(c_pthread_t), intent(in), value :: thread
            type(c_ptr),       intent(in)        :: value_ptr
            integer(kind=c_int)                  :: c_pthread_join
        end function c_pthread_join
    end interface
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

contains_template = """
CONTAINS

RECURSIVE SUBROUTINE kernel_ (errand_tid) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING 
    USE global, ONLY : errand_thread_state
    {argimport}
    IMPLICIT NONE
    TYPE(c_ptr), INTENT(IN), value :: errand_tid
    INTEGER, POINTER :: ERRAND_GOFER_ID_PTR
    INTEGER :: ERRAND_GOFER_ID

    {body}

    errand_thread_state(ERRAND_GOFER_ID) = 2

END SUBROUTINE
"""

fortran_modproc_template = """
INTEGER FUNCTION argtid (ptr)
    USE, INTRINSIC :: ISO_C_BINDING 
    TYPE(c_ptr), INTENT(IN), value :: ptr
    INTEGER, POINTER :: tid
    CALL c_f_pointer(ptr, tid)
    argtid = tid
END FUNCTION
"""

calldevmain_template = """
    USE global, ONLY : errand_threads, errand_thread_state, c_pthread_create
    IMPLICIT NONE
    INTEGER i, rc
    INTEGER, TARGET :: tids({nthreads}) = [ (i, i = 1, {nthreads}) ]

    ALLOCATE(errand_threads({nthreads}))
    ALLOCATE(errand_thread_state({nthreads}))

    DO i=1, {nthreads}

        errand_thread_state(i) = 0

        rc = c_pthread_create(thread=errand_threads(i), attr=c_null_ptr, &
                start_routine=c_funloc(kernel_), arg=c_loc(tids(i)))

        IF (rc .NE. 0) THEN
            errand_thread_state(i) = -1
        END IF

    END DO

    DO i=1, {nthreads}
        DO WHILE (errand_thread_state(i) .EQ. 0)
            CONTINUE
        END DO
    END DO
"""

class PThreadFortranBackend(FortranBackendBase):

    name = "pthread-fortran"
    codeext = "f90"
    libext = "so"

    def __init__(self, workdir, compile, **kwargs):

        compilers = Compilers(self.name, compile)
        targetsystem = select_system("cpu")

        super(PThreadFortranBackend, self).__init__(workdir, compilers,
            targetsystem, **kwargs)

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

    def num_threads(self):

        return numpy.prod(self.nteams) * numpy.prod(self.nmembers)

    def get_numpyattrs(self, arg):
        data = arg["data"]

        return ((data.ndim, data.itemsize, data.size) + data.shape +
                tuple([int(s//data.itemsize) for s in data.strides]))

    def code_attrtype(self):

        return fortran_attrtype_template

    def code_attrproc(self):

        return fortran_attrproc_template

    def code_modproc(self):

        return fortran_modproc_template

    def code_varattr(self):

        data = []

        for arg in self.inargs+self.outargs:

            data.append(arg["curname"])
            data.append(arg["curname"]+"_attr")

        return ",".join(data)

    def code_typedef(self):

        out = []

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)
            out.append("%s * %s;" % (self.getname_vartype(arg, "host"),
                        self.getname_var(arg, "host")))

        #out.append("int tid;")

        return typedef_template.format(args="\n".join(out))

    def code_attrdef(self):

        out = ""

        for arg in self.inargs+self.outargs:

            out += "CLASS(attrtype), ALLOCATABLE :: %s\n" % (arg["curname"]+"_attr")

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

    def code_modvardef(self):

        out = """
        TYPE(c_pthread_t), DIMENSION(:), ALLOCATABLE :: errand_threads
        INTEGER, DIMENSION(:), ALLOCATABLE :: errand_thread_state
"""

        return out

    def code_contains(self):

        argdef = []
        argassign = []

        goferid = ["IF (.NOT. C_ASSOCIATED(errand_tid)) RETURN",
                   "CALL c_f_pointer(errand_tid, ERRAND_GOFER_ID_PTR)",
                   "ERRAND_GOFER_ID = ERRAND_GOFER_ID_PTR",
                   "errand_thread_state(ERRAND_GOFER_ID) = 1"]

        section = str(self.order.get_section(self.name))
        bodylines = section.split("\n")
        firstexec = get_firstexec(bodylines)
        body = bodylines[:firstexec] + goferid + bodylines[firstexec:]

        arglist = []
        for arg in self.inargs+self.outargs:
            arglist.append(arg["curname"])
            arglist.append(arg["curname"]+"_attr")

        argimport = ""
        if arglist:
            argimport = "USE global, ONLY : " + ", ".join(arglist)

        return contains_template.format(argimport=argimport, body="\n".join(body))

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
                    varname=arg["curname"], attrname=arg["curname"]+"_attr",
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
                    varname=arg["curname"], attrname=arg["curname"]+"_attr",
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
                    varname=arg["curname"], attrname=arg["curname"]+"_attr",
                    bound=",".join(bound))

        return out

 
    def code_calldevmain(self):

        #body = str(self.order.get_section(self.name))

        return calldevmain_template.format(nthreads=self.num_threads())


    def code_stopbody(self):

        out = """
        USE global, ONLY : errand_threads, c_pthread_join, errand_threads, errand_thread_state
        IMPLICIT NONE
        INTEGER i, rc
        INTEGER, TARGET :: dummy

        DO i=1, {nthreads}
            rc = c_pthread_join(errand_threads(i), c_loc(dummy))
        END DO

        DEALLOCATE(errand_threads)
        DEALLOCATE(errand_thread_state)
""".format(nthreads=self.num_threads())

        return out

    def code_isbusybody(self):

        out = """
        USE global, ONLY : errand_thread_state
        IMPLICIT NONE
        INTEGER i

        DO i=1, {nthreads}
            IF (errand_thread_state(i) .GE. 0 .AND. errand_thread_state(i) .LT. 2) THEN
                isbusy = 1
                RETURN
            END IF
        END DO

        isbusy = 0
""".format(nthreads=self.num_threads())

        return out

    def get_template(self, name):

        if name == "h2dcopy":
            return pthrd_h2dcopy_template

        elif name == "h2dmalloc":
            return pthrd_h2dmalloc_template

        elif name == "d2hcopy":
            return pthrd_d2hcopy_template
