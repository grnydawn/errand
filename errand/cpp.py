"""Errand C++ and PThread C++ backend module


"""

import os
import numpy

from errand.backend import CppBackendBase, cpp_varclass_template
from errand.compiler import Compilers
from errand.system import select_system
from errand.util import which


host_vardef_template = """
{vartype} {varname} = {vartype}();
"""

pthrd_h2dcopy_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {varname}.data = ({dtype} *) data;
    {varname}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({varname}._attrs, _attrs, attrsize * sizeof(int));

    return 0;
}}
"""

pthrd_h2dmalloc_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {varname}.data = ({dtype} *) data;
    {varname}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({varname}._attrs, _attrs, attrsize * sizeof(int));

    return 0;
}}
"""

pthrd_d2hcopy_template = """
extern "C" int {name}(void * data) {{

    return 0;
}}
"""

devfunc_template = """
void * _kernel(void * ptr){{

    int ERRAND_GOFER_ID = *((int *)ptr);

    errand_thread_state[ERRAND_GOFER_ID] = 1;

    {body}

    errand_thread_state[ERRAND_GOFER_ID] = 2;

    return NULL;
}}
"""

calldevmain_template = """

    int tids[{nthreads}];

    for (int i=0; i < {nthreads}; i++) {{

        errand_thread_state[i] = 0;
        tids[i] = i;

        if (pthread_create(&(errand_threads[i]), NULL, _kernel, &(tids[i]))) {{
            errand_thread_state[i] = -1;
        }}
    }}

    for (int i=0; i < {nthreads}; i++) {{

        while (errand_thread_state[i] == 0) {{
            do {{ }} while(0);
        }}
    }}
"""

stopbody_template = """
    for (int i=0; i < {nthreads}; i++) {{
        pthread_join(errand_threads[i], NULL);
    }}

    free(errand_threads);
    free(errand_thread_state);

"""

isbusybody_template = """
    for (int i=0; i < {nthreads}; i++) {{
        if (errand_thread_state[i] >= 0 && errand_thread_state[i] < 2)
            return 1;
    }}

    return 0;
"""


class PThreadCppBackend(CppBackendBase):

    name = "pthread-c++"
    libext = "so"

    def __init__(self, workdir, compile, debug=0):

        self._debug = debug

        compilers = Compilers(self.name, compile)
        targetsystem = select_system("cpu")

        super(PThreadCppBackend, self).__init__(workdir, compilers,
            targetsystem)

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

    def num_threads(self):
        return numpy.prod(self.nteams) * numpy.prod(self.nmembers)

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
                dvsd[ndim] = cpp_varclass_template.format(vartype=hvartype, oparg=oparg,
                        offset=offset, funcprefix="", dtype=dname,
                        attrsize=attrsize)

        return "\n".join([y for x in dvs.values() for y in x.values()])


    def code_prerun(self):
    
        nthreads = self.num_threads()
        out = """
errand_threads = (pthread_t *) malloc(sizeof(pthread_t) * {nthreads});
errand_thread_state = (int *) malloc(sizeof(int) * {nthreads});
""".format(nthreads=nthreads)

        return out

    def code_vardef(self):

        out = """
pthread_t * errand_threads;
int * errand_thread_state;
"""

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            out += host_vardef_template.format(vartype=self.getname_vartype(arg,
                    "host"), varname=arg["curname"])

        return out

    def code_devfunc(self):

        body = str(self.order.get_section(self.name))

        return devfunc_template.format(body=body)

    def code_h2dcopyfunc(self):

        out = ""

        for arg in self.inargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dcopy(arg)

            template = self.get_template("h2dcopy")
            #hvar = self.getname_var(arg, "host")
            out += template.format(varname=arg["curname"], name=fname, dtype=dname)

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_h2dmalloc(arg)

            template = self.get_template("h2dmalloc")
            #hvar = self.getname_var(arg, "host")
            out += template.format(varname=arg["curname"], name=fname, dtype=dname)

        return out

    def code_d2hcopyfunc(self):

        out  = ""

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)
            fname = self.getname_d2hcopy(arg)

            template = self.get_template("d2hcopy")
            #hvar = self.getname_var(arg, "host")
            out += template.format(varname=arg["curname"], name=fname, dtype=dname)

        return out

 
    def code_calldevmain(self):

        nthreads = self.num_threads()
        return calldevmain_template.format(nthreads=str(nthreads))

    def code_stopbody(self):

        nthreads = self.num_threads()
        return stopbody_template.format(nthreads=str(nthreads))

    def code_isbusybody(self):

        nthreads = self.num_threads()
        return isbusybody_template.format(nthreads=str(nthreads))

    def get_template(self, name):

        if name == "h2dcopy":
            return pthrd_h2dcopy_template

        elif name == "h2dmalloc":
            return pthrd_h2dmalloc_template

        elif name == "d2hcopy":
            return pthrd_d2hcopy_template


class CppBackend(PThreadCppBackend):

    name = "c++"

    def num_threads(self):
        return 1
