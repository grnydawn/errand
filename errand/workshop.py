"""Errand workshop module


"""

import time

from collections import OrderedDict

from errand.util import split_compile
from errand.backend import select_backends

class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, workdir, backend=None, compile=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.compile = split_compile(compile)
        self.backends = select_backends(backend, self.compile, self.order, workdir)
        self.curbackend = None
        self.workdir = workdir
        self.code = None

    def start_backend(self, backend, nteams, nmembers, nassigns):

        self.code = backend.gencode(nteams, nmembers, nassigns, self.inargs,
                        self.outargs, self.order)

        backend.h2dcopy(self.inargs, self.outargs)

        res = self.code.run()

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
                print("backend '%s' is not workiing." % backend.name)
                # try multiple kinds of multiple backends
                pass

        raise Exception("No backend started.")

    # assumes that code.run() is async
    def close(self, timeout=None):

        if self.code is None:
            raise Exception("No code is generated.")

        while self.code.isalive() == 0 and (timeout is None or
            time.time()-self.start < float(timeout)):

            time.sleep(0.1)

        if self.curbackend is None:
            raise Exception("No selected backend")

        res = self.curbackend.d2hcopy(self.outargs)

        return res
