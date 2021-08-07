"""Errand test file"""

import os, numpy as np
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))

def test_vecadd1d():

    N = 100

    a = np.ones(N)
    b = np.ones(N) * 2
    c = np.zeros(N)

    order = os.path.join(here, "res", "vecadd.ord")

    # incorporate other models like kokkos and raja
    # kokkos: parallel_for and parallel_reduction, engine=execution space, numpy array = view
    # raja: indexset: launch configuration

    # include attrs of workshop like engine, ncores, nthreads/core, memory, ...
    # hip or cuda engine
    #with Errand(order, engine="hip") as erd:
    with Errand(order, timeout=10) as erd:

        # hierachical settings: order -> context -> base
        # best-effort of guessing default settings
        # eboys: one eboy is a thread(can be grouped multiple level, group, company, ...)
        # order: things to do(can have specific settings with multiple versions)
        # data: workload(managed between host-device, splitted and assigned to eboy)
        # engine is a hidden to user, that provides common interface to devices
        # context brings magic to user with coordination of eboys, order, and data with engine
        # KOKKOS is highly dependent on the concept of workload (iterations) than data view

        # logical worker entities
        gofers = erd.gofers(N)

        # workshop represents machine, input&output, order, and engine
        workshop = erd.workshop(a, b, "->", c)

        # generate source code, compile, and run
        gofers.run(workshop)
 
        # do whatever before gofers complete errand

    # When context complets, it ensures that errand is completed

    assert c.sum() == a.sum() + b.sum()
    #assert reduced_c == a.sum() + b.sum()

def ttest_vecadd2d():

    N1, N2 = 2, 3

#    a = np.ones((N1, N2))
#    b = np.ones((N1, N2)) * 2
#    c = np.zeros((N1, N2))


    a = np.arange(N1*N2).reshape(N1, N2)
    b = np.arange(N1*N2).reshape(N1, N2) * 2
    c = np.zeros((N1, N2))

    import pdb; pdb.set_trace()

    order = os.path.join(here, "res", "matadd.ord")

    with Errand(order) as erd:
        gofers = erd.gofers(N1, N2)

        workshop = erd.workshop(a, b, "->", c)

        gofers.run(workshop)

    assert c.sum() == a.sum() + b.sum()
