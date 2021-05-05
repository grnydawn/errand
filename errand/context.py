"""Errand Context module"""
 
from functools import partial
from numpy import ndarray


class _Context(object):

    def __init__(self, launcher):
        self._launcher = launcher

    def __call__(self, *vargs, **kwargs):

        # TODO: find out from ndarray shape of vargs
        launch_config = (1,)

        for arg in vargs:
            if isinstance(arg, ndarray):
                launch_config = arg.shape
                break

        pfunc = partial(self._launcher, config=launch_config)

        return pfunc(*vargs, **kwargs)

    def __getitem__(self, indices):

        if isinstance(indices, int):
            indices = (indices,)
        
        return partial(self._launcher, config=indices)


class Context(object):

    def __init__(self, esf, engine):

        self.esf = esf
        self.engine = engine
        self.run = _Context(self._kernel_launch)

    def _kernel_launch(self, *vargs, **kwargs):

        return self.engine.kernel_launch(self.esf, kwargs["config"], *vargs)

    def finish(self):
        pass

    def memcpy2device(self, *vargs, **kwargs):
        pass

    def memcpy2host(self, *vargs, **kwargs):
        pass

