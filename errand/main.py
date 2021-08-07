"""Errand main module

"""

# TODO: supports eval, attributes in order
# TODO: supports compiler hierachies with compiler wrappers

import shutil, tempfile

from errand.context import Context


class Errand(object):
    """Errand class

* Out of context, OS-related tasks here
"""

    def __init__(self, order, **kwargs):

        self.order = order
        self.kwargs = kwargs
 
        self.tempdir = None
        self.context = None
      
    def __enter__(self):

        self.tempdir = tempfile.mkdtemp()
        self.context =  Context(self.order, self.tempdir, **self.kwargs)

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.context:
            self.context.shutdown()

        if self.tempdir:
            shutil.rmtree(self.tempdir)

