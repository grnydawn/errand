"""Errand context module

Define Errand context

"""

from numpy import ndarray

class Context(object):
    """Context class: provides consistent interface of Errand

    * keep database
"""

    def __init__(self, order, engine, workdir):

        self.order = order
        self.engine = engine
        self.workdir = workdir

        self.devallocated = {}
        self.devcopied = {}
        self.inargs = []
        self.outargs = []

    def __getitem__(self, indices):
        # TODO: define launch configuration from indices

        return partial(self.run, config=indices)

    def run(self, *vargs, config=None, wait=False, copy2host=False):

        INARG = True
        devargs = []

        # allocated and copy data from host to gpu if needed
        for varg in vargs:

            if varg == "->":
                continue

            mid = id(varg)

            if mid not in self.devallocated:
                self.devmalloc(varg)

            if INARG:
                self.inarg.append[varg]

                if mid not in self.devcopied or not self.devcopied[mid]:
                    self.memcpy2dev(varg)

            else:
                self.outarg.append[varg]

        # launch GPU kernel
        self.engine.launch(*devargs, config=config, wait=wait)

        if copy2host:
            for oarg in self.outargs:
                self.memcpy2host(oarg)

    def shutdown(self):
        pass

    def devmalloc(self, var):

        if isinstance(var, ndarray):

            # (ctypes.c_ulong*3)()
            #a_p = a.ctypes.data_as(POINTER(c_float))
            import pdb; pdb.set_trace()
            self.devallocated[id(var)] = self.engine.devmalloc(var.nbytes)

        else:
            import pdb; pdb.set_trace()
