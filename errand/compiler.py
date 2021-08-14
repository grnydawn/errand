"""Errand compiler module

"""

import abc

class Compiler(abc.ABC):
    """Parent class for all compiler classes

"""
    compilers = {}

    def __init__(self):
        pass

def select_compiler(engine):

    if engine in Compiler.compilers:
        return Compiler.compilers[engine]()

    else:
        if Compiler.find(engine):
            return Compiler.compilers[engine]()

        else:
            raise Exception(("Compiler for '%s' is not available on this "
                    "system.") % engine)
