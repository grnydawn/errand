"""Errand workshop module


"""

import time

from collections import OrderedDict

from errand.backend import select_backend

class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, workdir, backend=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        backends = [b(workdir) for b in select_backend(backend, self.order)]
        self.backends = [b for b in backends if b.isavail()]
        self.curbackend = None
        self.workdir = workdir
        self.code = None

    def set_backend(self, backend):
        self.curbackend = backend

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

        try:

            if self.curbackend is not None:
                return self.start_backend(backend, nteams, nmembers, nassigns)

            else:
                for backend in self.backends:
                    return self.start_backend(backend, nteams, nmembers, nassigns)
 
        except Exception as e:
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
