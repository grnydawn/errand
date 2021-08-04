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

        # generate executable code
        self.code = self.engine.gencode(nteams, nmembers, self.inargs,
                        self.outargs, self.order)

        self.engine.h2dcopy(self.inargs, self.outargs)

        return self.code.run()

    def close(self):

        start = time.time()

        while self.code.isalive() == 0 and time.time()-start < 3:
            time.sleep(0.1)

        self.engine.d2hcopy(self.outargs)
