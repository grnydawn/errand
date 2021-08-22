"""Errand workshop module


"""

import time

from collections import OrderedDict


class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, engines, workdir):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.engines = engines
        self.curengine = None
        self.workdir = workdir
        self.code = None

    def set_engine(self, engine):
        self.curengine = engine

    def start_engine(self, engine, nteams, nmembers):

        self.code = engine.gencode(nteams, nmembers, self.inargs,
                        self.outargs, self.order)

        engine.h2dcopy(self.inargs, self.outargs)

        res = self.code.run()

        if res == 0:
            self.curengine = engine
            return res

        else:
            raise Exception("Engine is not started.") 


    def open(self, nteams, nmembers):

        self.start = time.time()

        try:

            if self.curengine is not None:
                return self.start_engine(engine, nteams, nmembers)

            else:
                for engine in self.engines:
                    return self.start_engine(engine, nteams, nmembers)
 
        except Exception as e:
            pass

        raise Exception("No engine started.")

    # assumes that code.run() is async
    def close(self, timeout=None):

        if self.code is None:
            raise Exception("No code is generated.")

        while self.code.isalive() == 0 and (timeout is None or
            time.time()-self.start < float(timeout)):

            time.sleep(0.1)

        if self.curengine is None:
            raise Exception("No selected engine")

        res = self.curengine.d2hcopy(self.outargs)

        return res
