"""Errand engine module


"""

import abc
from numpy.ctypeslib import ndpointer
from ctypes import c_int, c_double, c_size_t


_installed_engines = {}

class Engine(abc.ABC):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    def __init__(self, workdir):

        self.workdir = workdir
        self.kernel = None
        self.argmap = {}

    @classmethod
    @abc.abstractmethod
    def isavail(cls):
        pass

    @abc.abstractmethod
    def gencode(self, nteams, nmembers, inargs, outargs, order):
        pass

    def getname_h2dcopy(self, arg):

        name = self.argmap[id(arg)]
        return "h2dcopy_%s" % name

    def getname_d2hcopy(self, arg):

        name = self.argmap[id(arg)]
        return "d2hcopy_%s" % name

    def h2dcopy(self, inargs, outargs):

        for arg, attr in inargs+outargs:
            #np.ascontiguousarray(x, dtype=np.float32)
            h2dcopy = getattr(self.kernel, self.getname_h2dcopy(arg))
            h2dcopy.restype = c_int
            h2dcopy.argtypes = [ndpointer(c_double), c_size_t]

            res = h2dcopy(arg, arg.size)

    def d2hcopy(self, outargs):

        for arg, attr in outargs:
            d2hcopy = getattr(self.kernel, self.getname_d2hcopy(arg))
            d2hcopy.restype = c_int
            d2hcopy.argtypes = [ndpointer(c_double), c_size_t]

            res = d2hcopy(arg, arg.size)


class GpuEngine(Engine):
    pass

def select_engine(engine, order):

    if not _installed_engines:
        from errand.cuda import CudaEngine
        from errand.hip import HipEngine

        _installed_engines[CudaEngine.name] = CudaEngine
        _installed_engines[HipEngine.name] = HipEngine

    if isinstance(engine, Engine):
        return engine.__class__

    if isinstance(engine, str):
        if engine in _installed_engines:
            return _installed_engines[engine]
    else:
        for tname in order.get_targetnames():
            if tname in _installed_engines and _installed_engines[tname].isavail():
                return _installed_engines[tname]

    if engine is None:
        raise Exception("Engine-selection failed from the installed engines: %s"
                % ", ".join(_installed_engines.keys()))

    else:
        raise Exception("Engine-selection failed: %s" % str(engine))
