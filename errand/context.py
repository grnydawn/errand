"""Errand context module

Define Errand context

"""

import time, inspect, shutil, tempfile
from numpy import ndarray, asarray

from errand.order import Order
from errand.gofers import Gofers
from errand.workshop import Workshop

from errand.util import errand_builtins

# TODO: supports eval, attributes in order
# TODO: supports compiler hierachies with compiler wrappers


class Errand(object):
    """Errand class

* Out of context, OS-related tasks here
"""

    def __init__(self, order, timeout=None):

        self.tempdir = None
 
        # TODO: config data
        # TODO: documentation
        # TODO: examples
        # TODO: show cases(time slice to time series)
        # TODO: native programming support more than numpy-like arguments
        # TODO: timing measurement
        # TODO: compiler support
        # TODO: compiling cache
        # TODO: debugging support
        # TODO: logging support
        # TODO: testing support
        # TODO: optimization support
        # TODO: documentation support
        # TODO: plugin backend
        # TODO: registry for backends, orders, sharedlibs, etc.
        # TODO: order template generation for informing mapping from teams/gofers to language specfic interpretation, and data movements, and shared/private variables, ...
        # TODO: basic approaches: user focuses on computation. clear/simple/reasonable role of Errand

        self._env = dict(errand_builtins)
        self.workshops = {} # contains workshops
        self.result = [] # contains results from workshops

        self.order = order if isinstance(order, Order) else Order(order, self._env)

        self.timeout = timeout
     
    def __enter__(self):

        self.tempdir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        for ws in self.workshops:
            self.result.append(ws.close(timeout=self.timeout))

        if self.tempdir:
            shutil.rmtree(self.tempdir)

    def gofers(self, *vargs):

        # may have many optional arguments that hints
        # to determin how many gofers to be called, or the group hierachy 

        if len(vargs) > 0:
            return Gofers(*vargs)

        else:
            return Gofers(1)

    def _pack_argument(self, arg, caller_args):

        if isinstance(arg, ndarray):
            data = arg

        elif isinstance(arg, (list, tuple, set)):
            data = asarray(arg)

        elif isinstance(arg, dict):
            data = asarray([arg[k] for k in sorted(arg.keys())])

        else:
            # primitive types
            raise Exception("No supported type: %s" % str(type(arg)))

        name = caller_args[id(arg)]
        
        return {"data": data, "orgdata": arg, "npid": id(data),
                "memid": id(data.data), "orgname": name, "curname": name}

    def _split_arguments(self, vargs, caller_args):

        inargs = []
        outargs = None

        for varg in vargs:
            if isinstance(varg, str) and varg == "->":
                outargs = []
                continue

            if outargs is not None:

                if not isinstance(varg, (ndarray, list)):
                    raise Exception(("Output variable, '%s',"
                        "is not a numpy ndarray or list.") % caller_args[id(varg)])

                outargs.append(self._pack_argument(varg, caller_args))

            else:
                inargs.append(self._pack_argument(varg, caller_args))

        if outargs is None:
            outargs = []

        return (inargs, outargs)

    def workshop(self, *vargs, **kwargs):

        caller_local_vars = inspect.currentframe().f_back.f_locals.items()
        caller_args = dict([(id(v), n) for n, v in caller_local_vars])

        inargs, outargs = self._split_arguments(vargs, caller_args)

        ws = Workshop(inargs, outargs, self.order, self.tempdir, **kwargs)
        self.workshops[ws] = {}

        return ws
