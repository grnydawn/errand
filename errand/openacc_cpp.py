"""Errand OpenAcc backend module


"""

import numpy

from errand.cpp import CppBackend

devfunc_template = """
void * _kernel(void * ptr){{

    int ERRAND_GOFER_ID = *((int *)ptr);

    errand_thread_state[ERRAND_GOFER_ID] = 1;

#pragma acc enter data create({creates})
#pragma acc update device({dev_updates})

#pragma acc parallel num_gangs({ngangs}) num_workers({nworkers}) \
vector_length({veclen})
{{
    {body}
}}

#pragma acc update self ({host_updates})
#pragma acc exit data delete({deletes})

    errand_thread_state[ERRAND_GOFER_ID] = 2;

    return NULL;
}}
"""

class OpenAccCppBackend(CppBackend):

    name = "openacc-c++"


    def code_devfunc(self):

        creates = []
        deletes = []
        host_updates = []
        dev_updates = []

        body = str(self.order.get_section(self.name))

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

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

        gangs = numpy.prod(self.nteams)
        workers = numpy.prod(self.nmembers)
        veclen = numpy.prod(self.nassigns)

        return devfunc_template.format(body=body,
                    creates=", \\\n".join(creates),
                    dev_updates=", \\\n".join(dev_updates),
                    host_updates=", \\\n".join(host_updates),
                    deletes=", \\\n".join(deletes),
                    ngangs=str(gangs), nworkers=str(workers),
                    veclen=str(veclen))

