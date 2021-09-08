"""Errand workshop module


"""

import importlib, collections

from errand.util import split_compile

_supported_machmods = collections.OrderedDict({
    "hip": "hip",
})

_loaded_machmods = {
}


class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, backend=None, compile=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.backend = backend
        self.compile = split_compile(compile)
        self.machine = None

    def select_machine(self):

        order_backends = self.order.get_backends()

        if isinstance(self.backend, str):
            if self.backend not in _supported_machmods:
                raise Exception("No machine is available for '%s' backend" %
                                    self.backend)

            if self.backend not in order_backends:
                raise Exception("No order is avaiable for '%s' backend" %
                                    self.backend)

            modname = _supported_machmods[self.backend]

            if modname not in _loaded_machmods:
                _loaded_machmods[modname] = importlib.import_module(
                                                "errand."+modname)

            select = getattr(_loaded_machmods[modname], "select_machine", None)

            if select:
                for mach in select(self.compile, self.order): 
                    self.machine = mach

                    yield self.machine

        elif self.backend is None:
            for backend, modname in _supported_machmods.items():

                if backend not in order_backends:
                    continue

                if modname not in _loaded_machmods:
                    _loaded_machmods[modname] = importlib.import_module(
                                                    "errand."+modname)

                select = getattr(_loaded_machmods[modname], "select_machine",
                                    None)

                if select:
                    for mach in select(self.compile, self.order): 
                        self.machine = mach
                        self.backend = backend

                        yield self.machine

        else:
            raise Exception("backend argument should be a string type")

        if self.machine is None:
            raise Exception("No supported resource is available")

    def close(self):

        if self.machine:
            res = self.machine.unload(self.outargs)
            self.machine = None

