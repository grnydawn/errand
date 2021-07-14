"""Errand test file"""

import os, numpy as np
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))

def test_cuda():

    N = 10

    a = np.ones(N)
    b = np.ones(N)
    c = np.zeros(N)

    esf = os.path.join(here, "res", "vecadd.ord")

    # incorporate other models like kokkos and raja
    # kokkos: parallel_for and parallel_reduction, engine=execution space, numpy array = view
    # raja: indexset: launch configuration

    # include attrs of workshop like engine, ncores, nthreads/core, memory, ...
    with Errand(esf, engine="hip") as erd:

        # dmalloc occurs when needed
        #erd.dmalloc(a)
        #erd.dmalloc(b)
        #erd.dmalloc(c)

        # memcpy occurs when needed
        #erd.h2dmemcpy(a)
        #erd.h2dmemcpy(b)

        # user knows multicore and add HINTS as they know more
        # hints work only if available
        # others can be specified in erd order file
        # arrays are splitted per each thread so that each threads
        # sees only the part assigned to them
        # user sees data centri view - how to distribute my data to each core/threads
        # errand can dynamically use the # of cores and generate source code accordingly
        # KOKKOS is highly dependent on the concept of workload (iterations) than data view

        # split data and assign to block/threads
        # each threads run the assigned data and additional data

        # 1. create block/threads and assign to cores
        # 2. creates data and split and copy and assign to block/threads

        eboys = erd.call_eboys()
        sa, sb, sc = eboys.chop(a, b, c)
        eboys.run(sa, sb, sc)

# may order file handles reduce or loop
#        erd.parallel(a, b, "->", c)
#        erd.parallel_for(a, b, "->", c)
#        erd.parallel_reduce(a, b, "->", c)

        # memcpy occurs when needed
        #erd.d2hmemcpy(c)

        # free at exit automatically
        #erd.dfree(a)
        #erd.dfree(b)
        #erd.dfree(c)

    assert c.sum() == a.sum() + b.sum()
