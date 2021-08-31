"""Errand workshop module


"""

import abc

class Workshop(object):
    """Errand workshop class

"""

    def __next__(self):

        self.mach_index += 1

        if self.mach_index < len(self.machines):

            # build machine

            return machine

        raise StopIteration()

    def __iter__(self):
        return self

    def __init__(self, inargs, outargs, order, compile=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.compile = compile
        self.machines = self.select_machines(compile, order)
        self.mach_index = -1

    def select_machines(self, compile, order):

        machines = []

        for mach in MachineBase.__subclasses__():
            try:
                machines.append(mach(order=order, compile=compile))

            except:
                pass

        if len(machines) == 0:
            raise Exception("No supported resource is available")

        return machines

    def close(self):

        pass
        #res = self.curengine.d2hcopy(self.outargs)
        #return res

class MachineBase(abc.ABC):

    def __new__(cls, *vargs, **kwargs):

        obj = super(class_name, cls).__new__(cls, *vargs, **kwargs)

        if "order" not in kwargs:
            raise Exception("Can not find order input")

        obj.order = kwargs.pop("order")

        if "compile" in kwargs:
            compile = kwargs.pop("compile").lstrip()
            obj.compiler, flags = compile.split(" ", 1)
            obj.flags = cls.sharedlib_flags + " " + flags

        else:
            obj.compiler = cls.default_compiler
            obj.flags = cls.sharedlib_flags + " " + cls.default_flags

        obj.target = cls.ready_target()

        obj.check_avail()

        return obj

    def check_avail(self, compiler=None, flags=None, target=None):

        assert check_compiler(compiler if compiler else self.compiler) is True
        assert check_flags(flags if flags else self.flags) is True
        assert check_target(target if target else self.target) is True

    # raise exceptioin if fails
    @abc.abstractmethod
    def check_compiler(self, compiler):
        return False

    # raise exceptioin if fails
    def check_flags(self, compiler):
        return True

    # raise exceptioin if fails
    def check_target(self, compiler):
        return True

    @abc.abstractmethod
    def start(self, gofers):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def operate(self):
        pass

    @abc.abstractmethod
    def unload(self):
        pass

    @abc.abstractmethod
    def isbusy(self):
        return False

class CudaMachine(MachineBase):

    def check_compiler(self, compiler):
        return False

    def start(self, gofers):
        pass

    def load(self):
        pass

    def operate(self):
        pass

    def unload(self):
        pass

    def isbusy(self):
        return False
  

class HipMachine(MachineBase):
 
    def check_compiler(self, compiler):
        return False

    def start(self, gofers):
        pass

    def load(self):
        pass

    def operate(self):
        pass

    def unload(self):
        pass
   
    def isbusy(self):
        return False

class PthreadCppMachine(MachineBase):
 
    def check_compiler(self, compiler):
        return False

    def start(self, gofers):
        pass

    def load(self):
        pass

    def operate(self):
        pass

    def unload(self):
        pass
    
    def isbusy(self):
        return False
    
class OpenaccCppMachine(MachineBase):
  
    def check_compiler(self, compiler):
        return False

    def start(self, gofers):
        pass

    def load(self):
        pass

    def operate(self):
        pass

    def unload(self):
        pass
   
    def isbusy(self):
        return False

