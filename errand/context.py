"""Errand context module

Define Errand context

"""

import time

from errand.gofers import Gofers
from errand.workshop import Workshop


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

    def gofers(self, num_gofers=None):

        # may have many optional arguments that hints
        # to determin how many gofers to be called, or the group hierachy 
        # for now, let's call only one gofer

        if num_gofers is None:
            num_gofers = 1

        return Gofers(num_gofers)

    def workshop(self, *vargs, **kwargs):

        inargs = []
        outargs = None

        for varg in vargs:
            if varg == "->":
                outargs = []

            elif outargs is not None:
                outargs.append(varg)

            else:
                inargs.append(varg)

        ws = Workshop(inargs, outargs, self.order, self.engine, self.workdir, **kwargs)

        self.tasks[ws] = {}

        return ws

    def shutdown(self):

        for ws in self.tasks:
            self.output.append(ws.close())
