"""Errand workshop module


"""

import time

from collections import OrderedDict

from errand.util import split_compile
from errand.backend import select_backends

class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, workdir, debug=0, backend=0, compile=None):

        self._debug = 0
        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.compile = split_compile(compile)
        self.backends = select_backends(backend, self.compile, self.order, workdir, debug=debug)
        self.curbackend = None
        self.workdir = workdir
        self.code = None

    def start_backend(self, backend, nteams, nmembers, nassigns):

        self.code = backend.gencode(nteams, nmembers, nassigns, self.inargs,
                        self.outargs, self.order)

        backend.h2dcopy(self.inargs, self.outargs)

        res = backend.start()

        if res == 0:
            self.curbackend = backend
            return res

        else:
            raise Exception("Backend is not started.") 


    def open(self, nteams, nmembers, nassigns):

        self.start = time.time()

        for backend in self.backends:
            try:
                return self.start_backend(backend, nteams, nmembers, nassigns)
            except Exception as e:
                print("backend '%s' is not working." % backend.name)
                print(e)
                # try multiple kinds of multiple backends
                pass

        raise Exception("No backend started.")

    # assumes that code.start() is async
    def close(self, timeout=None):

        if self.curbackend is None:
            raise Exception("No selected backend")

        while self.curbackend.isbusy() != 0 and (timeout is None or
            time.time()-self.start < float(timeout)):

            time.sleep(0.1)

        if self._debug > 3:
            if timeout and float(timeout) < time.time()-self.start:
                print("Timeout occured.")

        res = self.curbackend.d2hcopy(self.outargs)

        self.curbackend.stop()

        return res
