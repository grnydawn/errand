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

    def get_typename(self):
        return self.ndarr.dtype.name

class OrderQueue(object):

    def __init__(self): 
        self.queue = []

    def append(self, order):
        self.queue.append(order)


class OrderPad(object):

    def __init__(self):

        self._queue = OrderQueue()
        self._mmap = {}
        self._argnames = []

    def get_argnames(self):
        return self._argnames

    def load_arguments(self, argnames, *args):

        isInput = True

        for aname, arg in zip(argnames, args):

            if arg == "->":
                isInput = False

                if arg != aname:
                    raise Exception("Non-matching argument: (%s, %s)" % (arg, aname))

                continue

            self._argnames.append(aname)

            if not isinstance(arg, ndarray):
                assert False, "TODO: handle non-ndarray variable"

            if aname in self._mmap:
                assert self._mmap[anme] is arg, "Mis-match argument: (%s, %s)" % (self._mmap[anme], arg)

            else:
                self._mmap[aname] = MemInfo(arg)
            
            minfo = self._mmap[aname]

            if isInput:
                minfo.set_input()

                if not minfo.is_dallocated():
                    self._queue.append(Order(Order.D_MALLOC, minfo))

                if not minfo.is_h2dcopied():
                    self._queue.append(Order(Order.H2D_MEMCPY, minfo))

            else:
                minfo.set_output()


    def get_vartype(self, arg, isInput):

        assert arg in self._mmap, "Argument '%s' is not in orderpad." % arg

        return self._mmap[arg].get_typename()


class OrderPads(object):

    DEFAULT_ORDERPAD = 0

    def __init__(self):

        self._orderpads = {self.DEFAULT_ORDERPAD: OrderPad()}
        self._current = self.DEFAULT_ORDERPAD

    def get_curorderpad(self):
        return self._orderpads[self.DEFAULT_ORDERPAD]
#        
#    def load_arguments(self, argnames, *args, orderpad=None):
#
#        if orderpad is None:
#            orderpad = self.DEFAULT_ORDERPAD
#
#        return self._orderpads[orderpad].load_arguments(argnames, *args)
#
#
#    def get_vartype(self, arg, isInput, orderpad=None):
#
#        if orderpad is None:
#            orderpad = self.DEFAULT_ORDERPAD
#
#        return self._orderpads[orderpad].get_vartype(arg, isInput)
#
