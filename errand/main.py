"""errand main module"""

import os

from errand.esf import SourceFile
from errand.engine import Engine, CUDAEngine
from errand.launcher import Launcher


class Errand(object):
    """
        Errand: T.B.D.
    """

    _name_ = "errand"
    _version_ = "0.1.0"
    _description_ = "Python GPU programming framework"
    _long_description_ = ("Python glue to multiple GPU "
                           "programming models in a source file")
    _author_ = "Youngsung Kim"
    _author_email_ = "youngsung.kim.act2@gmail.com"

    def __init__(self, path, engine=None):

        # load esf file

        if os.path.isfile(path):
            self.esf = SourceFile(path)

        else:
            raise Exception("Errand file is not found: %s" % str(path))

        # load gpu engine
        if engine is None:
            self.engine = self._choose_best_engine(self.esf)

        elif isinstance(engine, Engine):
            self.engine = engine

        elif isinstance(engine, str):
            self.engine = self._get_engine(engine)

        else:
            raise Exception("Wrong engine type: %s" % str(engine))

    def _get_engine(self, name):

        if name == "cuda":
            return CUDAEngine()

        else:
            raise Exception("Unknown GPU engine type: %s" % str(engine))
            
    def __enter__(self):

        self.launcher = Launcher(self.esf, self.engine)

        return self.launcher

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.launcher.finish()

