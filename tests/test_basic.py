"""Errand test file"""

import os, numpy as np
import pytest
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))
#test_engines = ["cuda", "hip", "pthread"]
test_engines = ["hip", "pthread", "openacc-c++"]
#test_engines = ["pthread"]
#test_engines = ["openacc-c++"]

@pytest.mark.parametrize("engine", test_engines)
def ttest_vecadd1d(engine):

    N = 100

    a = np.ones(N)
    b = np.ones(N) * 2
    c = np.zeros(N)

    order = os.path.join(here, "res", "vecadd1d.ord")

    # incorporate other models like kokkos and raja
    # kokkos: parallel_for and parallel_reduction, engine=execution space, numpy array = view
    # raja: indexset: launch configuration

    # include attrs of workshop like engine, ncores, nthreads/core, memory, ...
    # hip or cuda engine
    #with Errand(order, engine="hip") as erd:
    with Errand(order, engine=engine, timeout=5) as erd:

        # hierachical settings: order -> context -> base
        # best-effort of guessing default settings
        # eboys: one eboy is a thread(can be grouped multiple level, group, company, ...)
        # order: things to do(can have specific settings with multiple versions)
        # data: workload(managed between host-device, splitted and assigned to eboy)
        # engine is a hidden to user, that provides common interface to devices
        # context brings magic to user with coordination of eboys, order, and data with engine
        # KOKKOS is highly dependent on the concept of workload (iterations) than data view

        # logical worker entities
        # TODO: how to choose the best configuration per engine
        gofers = erd.gofers(N)

        # workshop represents machine, input&output, order, and engine
        workshop = erd.workshop(a, b, "->", c)

        # generate source code, compile, and run
        gofers.run(workshop)
 
        # do whatever before gofers complete errand

    # When context complets, it ensures that errand is completed

    assert c.sum() == a.sum() + b.sum()
    #assert reduced_c == a.sum() + b.sum()

@pytest.mark.parametrize("engine", test_engines)
def test_vecadd2d(engine):

    #NROW, NCOL = 2000, 300
    NROW, NCOL = 10, 3

    a = np.ones((NROW, NCOL))
    b = np.ones((NROW, NCOL))
    c = np.zeros((NROW, NCOL))

    order = os.path.join(here, "res", "vecadd2d.ord")

    with Errand(order, engine=engine, timeout=5) as erd:
        gofers = erd.gofers(NROW, NCOL)

        workshop = erd.workshop(a, b, "->", c)

        gofers.run(workshop)

    assert np.array_equal(c, a+b)

@pytest.mark.parametrize("engine", test_engines)
def test_listadd2d(engine):

    NROW = 2
    NCOL = 3

    a = [[1,1,1], [1,1,1]]
    b = [[1,1,1], [1,1,1]]
    c = [[0,0,0], [0,0,0]]

    order = os.path.join(here, "res", "vecadd2d.ord")

    with Errand(order, engine=engine, timeout=5) as erd:
        gofers = erd.gofers(NROW, NCOL)

        workshop = erd.workshop(a, b, "->", c)

        gofers.run(workshop)

    for i in range(NROW):
        for j in range(NCOL):
            assert c[i][j] == a[i][j] + b[i][j]
