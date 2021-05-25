"""errand main module"""

import os, tempfile, shutil

from errand.esf import SourceFile
from errand.engine import Engine, CUDAEngine
from errand.context import Context


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

    def __init__(self, path, engine=None, compiler=None):

        self.path = path
        self.engine = engine
        self.compiler = compiler

    def _get_engine(self, name):

        if name == "cuda":
            return CUDAEngine

        else:
            raise Exception("Unknown GPU engine type: %s" % str(name))
            
    def __enter__(self):

        self.tempdir = tempfile.mkdtemp()

        # load esf file
        if os.path.isfile(self.path):
            esf = SourceFile(self.path)
            assert esf is not None

            # load gpu engine
            engine_cls = None

            if self.engine is None:
                engine_cls = self._choose_best_engine(esf)

            elif isinstance(self.engine, str):
                engine_cls = self._get_engine(self.engine)

            elif issubclass(self.engine, Engine):
                engine_cls = self.engine

            elif not isinstance(self.engine, Engine):
                raise Exception("Wrong engine type: %s" % str(self.engine))

            assert engine_cls is not None

            engine = engine_cls(self.compiler, self.tempdir)

            self.ctx = Context(esf, engine)

            return self.ctx

        else:
            raise Exception("Errand file is not found: %s" % str(self.path))

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.ctx.finish()
        shutil.rmtree(self.tempdir)

