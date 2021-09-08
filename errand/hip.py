"Machines for Hip"


from errand.util import split_compile
from errand.machine import CudaHipBase
from errand.compile import HipccLinux

_name_ = "hip"

class HipMachine(CudaHipBase):
    pass


class LinuxHip(HipMachine):
    _compile_ = (
        HipccLinux,
    )

    def start(self, worker):
        pass

    def load(self, *inargs):
        pass

    def operate(self):
        pass

    def unload(self, *outargs):
        pass

    def isbusy(self):
        return False


def select_machine(compile, order):

    ord_sec = order.get_section(_name_)
    ord_args = order.get_argnames()
    ord_comp = split_compile(ord_sec.arg)

    if ord_sec is None:
        raise Exception("No backend for HIP in order file.")

    for mach in HipMachine.__subclasses__():
        for comp in mach.get_compilers(compile, ord_comp):
            yield mach(comp, ord_sec, ord_args)
