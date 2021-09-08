"""Errand compile module
"""

import re

pat_linuxhipcc = re.compile(r"^HIP\sversion:\s*\d+\.\d")

class Compiler(object):
    pass

class HipccLinux(Compiler):
    pass
    #"hipcc", "", "-fPIC --shared", pat_linuxhipcc),

def select_compiler(compile):
    # return object
    # select Compiler class
    # crate object with compile
    import pdb; pdb.set_trace()


