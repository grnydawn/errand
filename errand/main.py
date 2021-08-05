"""Errand main module

"""

import shutil, tempfile

from errand.order import Order
from errand.context import Context
from errand.engine import select_engine


class Errand(object):
    """Errand class

* manage temporary directory
* initialze errand activities
* finalize errand activities
  - wait for errand to finish
  - collect output
"""

    def __init__(self, order, engine=None, errand=None):

        self.order = Order(order)
        self.engine = engine
        self.errand = errand # to support hierachical errand
        
    def __enter__(self):

        self.tempdir = tempfile.mkdtemp()
        self.engine = select_engine(self.engine, self.order)(self.tempdir)
        self.context =  Context(self.order, self.engine, self.tempdir, self.errand)

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.context.shutdown()
        shutil.rmtree(self.tempdir)

