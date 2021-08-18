"""Errand compiler module

"""

import abc

from errand.util import which


class Compiler(abc.ABC):
    """Parent class for all compiler classes

"""

    def __init__(self, path):

        self.path = path

    def isavail(self):
        import pdb; pdb.set_trace()
        return self.path is not None and os.path.isfile(self.path)

    @abc.abstractmethod
    def get_option(self):
        pass

class C_Compiler(Compiler):

    def __init__(self, path):
        super(C_Compiler, self).__init__(path)


class CPP_Compiler(C_Compiler):

    def __init__(self, path):

        super(CPP_Compiler, self).__init__(path)


class GNU_C_Compiler(C_Compiler):

    def __init__(self, path=None):

        if path is None:
            path = which("gcc")

        super(GNU_C_Compiler, self).__init__(path)

    def get_option(self):
        return ""


class Compilers(object):

    def __init__(self, engine):

        if engine == "c":
            self.clist =  [GNU_C_Compiler()]

        elif engine == "c++":
            self.clist =  []

        else:
            raise Exception("Compiler for '%s' is not supported." % engine)

    def isavail(self):

        return self.select_one() is not None        

    def select_one(self):

        for comp in self.clist:
            if comp.isavail():
                return comp

    def select_many(self):

        comps = []

        for comp in self.clist:
            if comp.isavail():
                comps.append(comp)

        return comps

