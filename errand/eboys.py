"""Errand boys module


"""

import os, sys

from numpy import double
from numpy.ctypeslib import load_library, ndpointer
from threading import Thread


class EBoys(object):
    """Errand boys class

"""

    def __init__(self, num_eboys):

        self.num_eboys = num_eboys
        self.args = []
        self.sharedlib = None
        self.dismissed = False

    def load(self, data, assign_method=None):

        self.args.append(data)

    def run(self, lib):

        head, tail = os.path.split(lib)
        base, ext = os.path.splitext(tail)

        array_1d_double = ndpointer(dtype=double, ndim=1, flags='CONTIGUOUS')

        # load the library, using numpy mechanisms
        self.sharedlib = load_library(base, head)

        # setup the return types and argument types
        #libkernel.run.restype = None
        #libkernel.run.argtypes = [array_1d_double, array_1d_double, c_int]

        # launch cuda program
        th = Thread(target=self.sharedlib.run)
        th.start()

    def dismiss(self):

        if not self.dismissed:

            # signal device kernel to exit
            if self.sharedlib:
                self.sharedlib.stop()

            self.dismissed = True
