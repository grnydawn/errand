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


def select_system(name):

    if name == "cpu":
        return CPUSystem()

    else:
        raise Exception("Unknown system: %s" % name)
