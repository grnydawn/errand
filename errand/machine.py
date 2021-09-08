"""Errand machine module


"""

import abc

from errand.compile import select_compiler

class MachineBase(abc.ABC):

    def __new__(cls, compile, ordsec, ordarg):

        obj = super(MachineBase, cls).__new__(cls)

        obj.compile = compile
        obj.ordsec = ordsec 
        obj.ordarg = ordarg 
        obj.worker = None

        return obj

    @classmethod
    def get_compilers(cls, ws_compile, ord_compile):
        if ord_compile:
            return [select_compiler(ord_compile)]

        elif ws_compile:
            return [select_compiler(ws_compile)]

        else:
            for comp in cls._compile_:
                try:
                    yield comp()
                except Exception as err:
                    raise

    @abc.abstractmethod
    def start(self, worker):
        pass

    @abc.abstractmethod
    def load(self, *inargs):
        pass

    @abc.abstractmethod
    def operate(self):
        pass

    @abc.abstractmethod
    def unload(self, *outargs):
        pass

    @abc.abstractmethod
    def isbusy(self):
        return False

class CudaHipBase(MachineBase):
    pass
