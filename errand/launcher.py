"""Errand launcher module"""

class _Launcher(object):

    def __call__(self, *vargs, **kwargs):
        pass

    def __getitem__(self, indices):

        if isinstance(indices, int):

            return self._int_launcher
 
        else:
            raise Exception("Not supported indexing type YET: %s" %
                            str(indices))

    def _int_launcher(self, *vargs, **kwargs):
        pass

class Launcher(object):

    def __init__(self, esf, engine):

        self.esf = esf
        self.engine = engine
        self.run = _Launcher()

    def finish(self):
        pass

    def memcpy2device(self, *vargs, **kwargs):
        pass

    def memcpy2host(self, *vargs, **kwargs):
        pass

