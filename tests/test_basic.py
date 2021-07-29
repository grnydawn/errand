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
    # hip or cuda engine
    with Errand(esf, engine="cuda") as erd:

        # hierachical settings: order -> context -> base
        # best-effort of guessing default settings
        # eboys: one eboy is a thread(can be grouped multiple level, group, company, ...)
        # order: things to do(can have specific settings with multiple versions)
        # data: workload(managed between host-device, splitted and assigned to eboy)
        # engine is a hidden to user, that provides common interface to devices
        # context brings magic to user with coordination of eboys, order, and data with engine
        # KOKKOS is highly dependent on the concept of workload (iterations) than data view

        # call errand boys with optional hierachical groups
        eboys = erd.call_eboys()

        # assign workload to eboys/groups/eboy
        # may add reduce=method argument
        erd.assign(eboys, a, b, c)
        #erd.assign(eboys, a, b, c, reduce=erd.reduce.sum(c))

        # go!~
        erd.run(eboys)

        # gather result of the task
        erd.gather(eboys, c)
        #reduced_c = erd.gather(eboys, c) if reduce is used
 
    assert c.sum() == a.sum() + b.sum()
    #assert reduced_c == a.sum() + b.sum()
