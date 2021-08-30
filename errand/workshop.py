"""Errand workshop module


"""


class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, method=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.method = method
        self.engine = None

    def open(self, engine, nteams, nmembers, nassigns):

        for engine in self.select_engines(self.method):
            try:
                engine.start(nteams, nmembers, nassigns, self.inargs,
                        self.outargs, self.order)
                self.engine = engine

            except Exception as err:
                pass

    def ready(self):

        if self.engine:
            # copy data to target machine
        else:
            raise Exception("No engine is started.")


    def run(self):
        pass

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
