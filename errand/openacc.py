"""Errand OpenAcc engine module


"""

import os

from errand.engine import Engine, varclass_template
from errand.compiler import Compilers
from errand.system import select_system
from errand.util import which


struct_template = """
typedef struct arguments {{
    {args}
}} ARGSTYPE;

typedef struct wrap_args {{
    ARGSTYPE * data;
    int tid;
    int state;
}} WRAPARGSTYPE;
"""

host_vardef_template = """
{vartype} {varname} = {vartype}();
"""

varglobal_template = """
ARGSTYPE struct_args = {{
{varassign}
}};
"""

h2dcopy_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    return 0;
}}
"""

h2dmalloc_template = """
extern "C" int {name}(void * data, void * _attrs, int attrsize) {{

    {hvar}.data = ({dtype} *) data;
    {hvar}._attrs = (int *) malloc(attrsize * sizeof(int));
    memcpy({hvar}._attrs, _attrs, attrsize * sizeof(int));

    return 0;
}}
"""

d2hcopy_template = """
extern "C" int {name}(void * data) {{

    return 0;
}}
"""

devfunc_template = """
void * _kernel(void * ptr){{
    {argdef}

    WRAPARGSTYPE * args = (WRAPARGSTYPE *)ptr;

    args->state = 1;

    {argassign}

#pragma acc enter data create({creates})
#pragma acc update device({dev_updates})

#pragma acc parallel num_gangs({ngangs}) num_workers({nworkers}) \
vector_length({veclen})
{{
    {body}
}}

#pragma acc update self ({host_updates})
#pragma acc end data delete({deletes})

    args->state = 2;

    isfinished = 1;
}}
"""

calldevmain_template = """

    pthread_t thread;
    WRAPARGSTYPE args;

    pthread_attr_t attr;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    args.tid = 0;
    args.state = 0;
    args.data = &struct_args;

    pthread_create(&thread, &attr, _kernel, &args);

    while (args.state == 0) {{
        do {{ }} while(0);
    }}

"""

class OpenAccCppEngine(Engine):

    name = "openacc-c++"
    codeext = "cpp"
    libext = "so"

    def __init__(self, workdir):

        compilers = Compilers(self.name)
        targetsystem = select_system("cpu")

        super(OpenAccCppEngine, self).__init__(workdir, compilers,
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
                dvsd[ndim] = varclass_template.format(vartype=hvartype, oparg=oparg,
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

    def code_devfunc(self):

        argdef = []
        argassign = []
        creates = []
        deletes = []
        host_updates = []
        dev_updates = []

        body = str(self.order.get_section(self.name))

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            #argdef.append("host_%s_dim%s %s = host_%s_dim%s();" %
            #        (dname, ndim, arg["curname"], dname, ndim))
            argdef.append("host_%s_dim%s %s;" %
                    (dname, ndim, arg["curname"]))
            argassign.append("%s = *(args->data->host_%s);" %
                    (arg["curname"], arg["curname"]))
            accstr = ("{name}.data[0:{name}._attrs[2]], "
                "{name}._attrs[0:{name}._attrs[2]]").format(name=arg["curname"])
            creates.append(accstr)
            deletes.append("{name}.data, {name}._attrs".format(name=arg["curname"]))

        for arg in self.inargs:

            ndim, dname = self.getname_argpair(arg)

            accstr = ("{name}.data[0:{name}._attrs[2]], "
                "{name}._attrs[0:{name}._attrs[2]]").format(name=arg["curname"])

            dev_updates.append(accstr)

        for arg in self.outargs:

            ndim, dname = self.getname_argpair(arg)

            host_updates.append("{name}.data[0:{name}._attrs[2]]".
                format(name=arg["curname"]))

        return devfunc_template.format(argdef="\n".join(argdef), body=body,
                    argassign="\n".join(argassign),
                    creates=", \\\n".join(creates),
                    dev_updates=", \\\n".join(dev_updates),
                    host_updates=", \\\n".join(host_updates),
                    deletes=", \\\n".join(deletes),
                    ngangs=str(self.nteams), nworkers="1",
                    veclen=str(self.nmembers))

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

        return calldevmain_template.format(
                nthreads=str(self.nteams * self.nmembers))

    def get_template(self, name):

        if name == "h2dcopy":
            return h2dcopy_template

        elif name == "h2dmalloc":
            return h2dmalloc_template

        elif name == "d2hcopy":
            return d2hcopy_template
