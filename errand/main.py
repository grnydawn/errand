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
  - load order file
  - load gpu engine
* finalize errand activities
  - unload gpu engine
"""

    def __init__(self, order, engine=None):

        self.order = Order(order)
        self.engine = select_engine(engine, self.order)
        
    def __enter__(self):

        self.tempdir = tempfile.mkdtemp()
        self.context =  Context(self.order, self.engine, self.tempdir)

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.context.shutdown()
        shutil.rmtree(self.tempdir)

