"""Errand OpenAcc backend module


"""

import numpy

from errand.cpp import CppBackend

calldevmain_template = """


#pragma acc enter data copyin({copyin})
#pragma acc parallel async num_gangs({ngangs}) num_workers({nworkers}) \
vector_length({veclen})
{{
    {body}
}}

"""

stopbody_template = """

#pragma acc wait
#pragma acc exit data copyout({copyout})
"""

class OpenAccCppBackend(CppBackend):

    name = "openacc-c++"
   

    def code_devfunc(self):
        return ""

    def code_vardef(self):

        out = ""

        for arg in self.inargs+self.outargs:

            ndim, dname = self.getname_argpair(arg)

            out += "{vartype} {varname} = {vartype}();\n".format(vartype=self.getname_vartype(arg,
                    "host"), varname=arg["curname"])

        return out

    def code_isbusybody(self):
        return "return 0;"

    def code_prerun(self):
        return ""

    def code_stopbody(self):

        copyout = []
        deletes = []
        host_updates = []

#        for arg in self.inargs+self.outargs:
#            deletes.append("{name}.data, {name}._attrs".format(name=arg["curname"]))

        for arg in self.outargs:

#            ndim, dname = self.getname_argpair(arg)
#
#            host_updates.append("{name}.data[0:{name}._attrs[2]]".
#                format(name=arg["curname"]))

            copyout.append(arg["curname"])

        return stopbody_template.format(copyout=", ".join(copyout))

    def code_calldevmain(self):

        copyin = []

        body = str(self.order.get_section(self.name))

        for arg in self.inargs+self.outargs:

#            ndim, dname = self.getname_argpair(arg)
#
#            copyin.append(arg["curname"])
#            accstr = ("{name}.data[0:{name}._attrs[2]], "
#                "{name}._attrs[0:{name}._attrs[2]]").format(name=arg["curname"])
#            creates.append(accstr)
            copyin.append(arg["curname"])

#        for arg in self.outargs:
#
#            ndim, dname = self.getname_argpair(arg)
#
#            accstr = ("{name}.data[0:{name}._attrs[2]], "
#                "{name}._attrs[0:{name}._attrs[2]]").format(name=arg["curname"])
#
#            dev_updates.append(accstr)

        gangs = numpy.prod(self.nteams)
        workers = numpy.prod(self.nmembers)
        veclen = numpy.prod(self.nassigns)

        return calldevmain_template.format(body=body,
                    copyin=", ".join(copyin),
                    ngangs=str(gangs), nworkers=str(workers),
                    veclen=str(veclen))

