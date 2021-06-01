"""Errand context module

Define Errand context

"""


class Context(object):
    """Context class: provides consistent interface of Errand

    * keep database
"""

    def __init__(self, order, engine, workdir):
        self.order = order
        self.engine = engine
        self.workdir = workdir

    def __getitem__(self, indices):
        # TODO: define launch configuration from indices

        return partial(self.run, config=indices)

    def run(self, *vargs, config=None):

        # allocated and copy data from host to gpu if needed

        # TODO: launch GPU kernel

        # copy data from gpu to host if needed and deallocate
        pass

    def shutdown(self):
        pass

class CudaContext(Context):
    pass

class HipContext(Context):
    pass

def select_context(order, engine, workdir):

    if engine.name == "cuda":
        return CudaContext(order, engine, workdir)

    elif engine.name == "hip":
        return HipContext(order, engine, workdir)

    else:
        raise Exception("Unknown context type: %s" % engine.name)
