"""Errand engine module


"""

import abc


class Engine(abc.ABC):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    def __init__(self, workdir):
        self.workdir = workdir

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

    if isinstance(engine, Engine):
        return engine.__class__

    if isinstance(engine, str):
        if engine == "cuda":
            from errand.cuda import CudaEngine
            return CudaEngine

        elif engine == "hip":
            from errand.hip import HipEngine
            return HipEngine

        else:
            raise Exception("Not supported engine type: %s" % engine)

    elif order:
        raise Exception("Engine-selection from order is not supported")

    # TODO: auto-select engine from sysinfo
