"""Errand Context module"""
 
from functools import partial
from numpy import ndarray
from inspect import currentframe

from errand.orderpad import OrderPads

class _Context(object):

    def __init__(self, launcher):
        self._launcher = launcher

    def __call__(self, *vargs, **kwargs):

        # TODO: find out launch config from ndarray shape of vargs
        launch_config = (1,)

        argnames = []
        varids = {}

        for n, v in currentframe().f_back.f_locals.items():
            varids[id(v)] = n

        for varg in vargs:
            if varg == "->":
                argnames.append(varg)

            else:
                vid = id(varg)
                if vid in varids:
                    argnames.append(varids[vid])

        for arg in vargs:
            if isinstance(arg, ndarray):
                launch_config = arg.shape
                break

        pfunc = partial(self._launcher, config=launch_config, argnames=argnames)

        return pfunc(*vargs, **kwargs)

    def __getitem__(self, indices):

        if isinstance(indices, int):
            indices = (indices,)
        
        return partial(self._launcher, config=indices)


#class _VarMap(object):
#
#    def __init__(self):
#
#        self._varmap = {} # H:D
#
#    def H2D(self, *vargs, **kwargs):
#
#        for varg in vargs:
#            if varg not in self._varmap:
#                if isinstance(varg, ndarray):
#                    self._varmap[varg] = None
#
#                else:
#                    import pdb; pdb.set_trace()
#
#            if self._varmap[varg] is None:
#                import pdb; pdb.set_trace()
#
#    def D2H(self, *vargs, **kwargs):
#
#        for varg in vargs:
#            import pdb; pdb.set_trace()

class Context(object):

    def __init__(self, esf, engine):

        self.esf = esf
        self.engine = engine
        self.orderpads = OrderPads()
        self.run = _Context(self._kernel_launch)

    def _kernel_launch(self, *vargs, **kwargs):

        return self.engine.kernel_launch(self.orderpads, self.esf,
            kwargs["config"], kwargs["argnames"], *vargs)

    def finish(self):
        pass

