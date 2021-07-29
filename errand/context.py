"""Errand context module

Define Errand context

"""

import time

from errand.eboys import EBoys
from errand.data import ManagedData

from numpy import ndarray

# TODO: Context is reposible to provide a concept of Errand for use

class Context(object):
    """Context class: provides consistent interface of Errand

    * keep database
"""

    def __init__(self, order, engine, workdir, context):

        self.egroups = {}

        self.order = order # what eboys to do
        self.engine = engine # what eboys to use to complete the order
        self.workdir = workdir
        self.within = context

        # context should know all of current states

    def call_eboys(self, num_eboys=None):

        # may have many optional arguments that hints
        # to determin how many eboys to be called, or the group hierachy 
        # for now, let's call only one eboy

        if num_eboys is None:
            num_eboys = 1

        eboys = EBoys(num_eboys)
        self.egroups[eboys] = {}

        return eboys

    def assign(self, eboys, *vargs, method=None):

        for varg in vargs:
            if not isinstance(varg, ManagedData):
                varg = ManagedData(varg)

            eboys.load(varg, assign_method=method)

    def run(self, eboys):

        # TODO: generate code
        code = self.engine.gen_code(self.order, self.workdir)

        # TODO: compile code
        lib = self.engine.gen_sharedlib(code, self.workdir)

        # TODO: assign it to eboys
        eboys.run(lib)

    def gather(self, eboys, data, *vargs, tolist=False):

        out = []

        out.append(self._gather(eboys, data))

        for varg in vargs:
            out.append(self._gather(eboys, varg))
            
        return out if tolist or vargs else out[0]

    def _gather(self, eboys, data):
        pass

    def dismiss(self, *vargs):

        for eboys in vargs:
            eboys.dismiss()

    def shutdown(self):

        for eboys, cfg in self.egroups.items():
            self.dismiss(eboys)
