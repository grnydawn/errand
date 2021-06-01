"""Errand accellerator engine module


"""


class Engine(object):
    """Errand Engine class

    * keep as transparent and passive as possible
"""

    pass


class CudaEngine(Engine):

    name = "cuda"



class HipEngine(Engine):

    name = "hip"


def select_engine(engine, order):

    if isinstance(engine, Engine):
        return engine

    if isinstance(engine, str):
        if engine == "cuda":
            return CudaEngine()

        elif engine == "hip":
            return HipEngine()

        else:
            raise Exception("Not supported engine type: %s" % engine)

    # TODO: auto-select engine from sysinfo
