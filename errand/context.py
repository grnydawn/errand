"""Errand context module

Define Errand context

"""

import time

from errand.gofers import Gofers
from errand.workshop import Workshop
from errand.util import split_arguments


class Context(object):
    """Context class: provides consistent interface of Errand

"""

    def __init__(self, order, engine, workdir, context):

        self.tasks = {}

        self.order = order
        self.engine = engine
        self.workdir = workdir
        self.context = context
        self.output = []

    def gofers(self, *vargs):

        # may have many optional arguments that hints
        # to determin how many gofers to be called, or the group hierachy 

        if len(vargs) > 0:
            return Gofers(*vargs)

        else:
            return Gofers(1)


    def workshop(self, *vargs, **kwargs):

        inargs, outargs = split_arguments(*vargs)

        ws = Workshop(inargs, outargs, self.order, self.engine, self.workdir, **kwargs)

        self.tasks[ws] = {}

        return ws

    def shutdown(self):

        for ws in self.tasks:
            self.output.append(ws.close())
