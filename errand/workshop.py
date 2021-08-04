"""Errand workshop module


"""

from collections import OrderedDict


class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs):

        self.inargs = [(i, {}) for i in inargs]
        self.outargs = [(o, {}) for o in outargs]

    def shutdown(self):
        pass
