"""Errand orderpad module"""

import ctypes
from numpy import ndarray

#E = ctypes.cdll.LoadLibrary("/sw/summit/cuda/10.1.243/lib64/libcudart.so")
# TODO: create a code that wraps cuda data movements
#       and create a so file from it and load here


#class VarType(object):
#    pass

# TODO: order in queue depends on other order such as kernel launch


class Order(object):

    D_MALLOC, H2D_MEMCPY = range(2)

    def __init__(self, ordertype, minfo):
        self.ordertype = ordertype
        self.minfo = minfo


class MemInfo(object):

    def __init__(self, arg):
        self.ndarr = arg
        self.intype = None
        self.outtype = None 
        self.dalloc = None
        self.h2dcopied = None
        self.d2hcopied = None

    def set_input(self, intype=True):
        self.intype = intype

    def set_output(self, outtype=True):
        self.outtype = outtype

    def is_dallocated(self):
        return self.dalloc is not None

    def is_h2dcopied(self):
        return self.h2dcopied is not None

    def is_d2hcopied(self):
        return self.d2hcopied is not None

class OrderQueue(object):

    def __init__(self): 
        self.queue = []

    def append(self, order):
        self.queue.append(order)


class OrderPad(object):

    def __init__(self):

        self._queue = OrderQueue()
        self._mmap = {}

    def load_arguments(self, *args):

        isInput = True

        for arg in args:

            if arg == "->":
                isInput = False
                continue

            if not isinstance(arg, ndarray):
                assert False, "TODO: handle non-ndarray variable"

            uid = id(arg)

            if uid not in self._mmap:
                self._mmap[uid] = MemInfo(arg)
            
            minfo = self._mmap[uid]

            if isInput:
                minfo.set_input()

                if not minfo.is_dallocated():
                    self._queue.append(Order(Order.D_MALLOC, minfo))

                if not minfo.is_h2dcopied():
                    self._queue.append(Order(Order.H2D_MEMCPY, minfo))

            else:
                minfo.set_output()


    def get_vartype(self, arg, isInput):

        uid = id(arg)
        import pdb; pdb.set_trace()
        assert uid in self._mmap, "Argument '%s' is not in orderpad." % arg

        import pdb; pdb.set_trace()


class OrderPads(object):

    DEFAULT_ORDERPAD = 0

    def __init__(self):

        self._orderpads = {self.DEFAULT_ORDERPAD: OrderPad()}

    def load_arguments(self, *args, orderpad=None):

        if orderpad is None:
            orderpad = self.DEFAULT_ORDERPAD

        return self._orderpads[orderpad].load_arguments(*args)


    def get_vartype(self, arg, isInput, orderpad=None):

        if orderpad is None:
            orderpad = self.DEFAULT_ORDERPAD

        return self._orderpads[orderpad].get_vartype(arg, isInput)

