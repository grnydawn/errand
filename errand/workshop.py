"""Errand workshop module


"""

import time

from collections import OrderedDict


class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, engine, workdir):

        self.inargs = [(i, {}) for i in inargs]
        self.outargs = [(o, {}) for o in outargs]

        self.order = order
        self.engine = engine
        self.workdir = workdir
        self.code = None

    def open(self, nteams, nmembers):

        self.start = time.time()

        # generate executable code
        self.code = self.engine.gencode(nteams, nmembers, self.inargs,
                        self.outargs, self.order)

        self.engine.h2dcopy(self.inargs, self.outargs)

        return self.code.run()

    # assumes that code.run() is async
    def close(self, timeout=None):

        while self.code.isalive() == 0 and (timeout is None or
            time.time()-self.start < float(timeout)):

            time.sleep(0.1)

        self.engine.d2hcopy(self.outargs)
