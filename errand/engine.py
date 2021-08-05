"""Errand engine module


"""

import abc


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

    @abc.abstractmethod
    def h2dcopy(self, inargs, outargs):
        pass

    @abc.abstractmethod
    def d2hcopy(self, inargs):
        pass


def select_engine(engine, order):

    if not _installed_engines:
        from errand.cuda import CudaEngine
        from errand.hip import HipEngine

        _installed_engines["cuda"] = CudaEngine
        _installed_engines["hip"] = HipEngine

    if isinstance(engine, Engine):
        return engine.__class__

    if isinstance(engine, str):
        if engine in _installed_engines:
            return _installed_engines[engine]
    else:
        for tname in order.get_targetnames():
            if tname in _installed_engines and _installed_engines[tname].isavail():
                return _installed_engines[tname]

    raise Exception("Engine-selection failed: %s" % str(engine))
