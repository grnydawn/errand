"""Errand workshop module


"""

import abc

class Workshop(object):
    """Errand workshop class

"""

    def __init__(self, inargs, outargs, order, compile=None):

        self.inargs = inargs
        self.outargs = outargs
        self.order = order
        self.compile = compile
        self.machines = self.select_machines(compile, order)
        self.mach_index = -1

    def __next__(self):

        self.mach_index += 1

        if self.mach_index < len(self.machines):

            # build machine

            return machine

        raise StopIteration()

    def __iter__(self):
        return self

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

        obj = super(MachineBase, cls).__new__(cls)

        if "order" not in kwargs:
            raise Exception("Can not find order input")

        obj.order = kwargs.pop("order")

        if "compile" in kwargs:
            compile = kwargs.pop("compile").lstrip()
            temp = compile.split(" ", 1)
            obj.compiler, flags = temp if len(temp)==2 else (compile, "")
            obj.flags = cls.sharedlib_flags + " " + flags

            check compiler

        else:
            for comp, sflags, flags, repat in cls.compile:
                try:
                    obj.compiler = cls.default_compiler
                    obj.flags = cls.sharedlib_flags + " " + cls.default_flags

                    check compiler
                except:

        obj.target = cls.ready_target()

        obj.check_avail()

        return obj

    def check_avail(self, compiler=None, flags=None, target=None):

        assert self.check_compiler(compiler if compiler else self.compiler) is True
        assert self.check_flags(flags if flags else self.flags) is True
        assert self.check_target(target if target else self.target) is True

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

    default_compiler = "nvcc"
    sharedlib_flags = "--compiler-options '-fPIC' --shared"
    default_flags = ""

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
 
    compiles = (
        ("hipcc", "-fPIC --shared", "", b"dsfdsfsd"),
    )

    def check_compiler(self, compiler):
        import pdb; pdb.set_trace()
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
 
    default_compiler = "nvcc"
    sharedlib_flags = ""
    default_flags = ""

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
  
    default_compiler = "nvcc"
    sharedlib_flags = ""
    default_flags = ""

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

