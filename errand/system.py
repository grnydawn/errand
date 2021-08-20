"""Errand System module


"""

import abc


class System(abc.ABC):
    """Errand system class

"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def isavail(self):
        pass


class CPUSystem(System):

    def isavail(self):
        return True

class NvidiaGPUSystem(System):

    def isavail(self):
        return True


class AmdGPUSystem(System):

    def isavail(self):
        return True


def select_system(name):

    if name == "cpu":
        return CPUSystem()

    elif name == "nvidia-gpu":
        return NvidiaGPUSystem()

    elif name == "amd-gpu":
        return AmdGPUSystem()

    else:
        raise Exception("Unknown system: %s" % name)
