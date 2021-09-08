"""Errand test file"""

import os, numpy as np
import pytest
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))
#test_backends = ["cuda", "hip", "pthread"]
#test_backends = ["hip", "pthread", "openacc-c++"]
test_backends = ["hip"]
#test_backends = ["pthread"]
#test_backends = ["openacc-c++"]

@pytest.mark.parametrize("backend", test_backends)
def test_vecadd1d(backend):

    N = 100

    a = np.ones(N)
    b = np.ones(N) * 2
    c = np.zeros(N)

    order = os.path.join(here, "res", "vecadd1d.ord")

    # incorporate other models like kokkos and raja
    # kokkos: parallel_for and parallel_reduction, compile=execution space, numpy array = view
    # raja: indexset: launch configuration

    # include attrs of workshop like compile, ncores, nthreads/core, memory, ...
    # hip or cuda compile
    #with Errand(order, compile="hip") as erd:
    with Errand(order, timeout=5) as erd:

        # hierachical settings: order -> context -> base
        # best-effort of guessing default settings
        # eboys: one eboy is a thread(can be grouped multiple level, group, company, ...)
        # order: things to do(can have specific settings with multiple versions)
        # data: workload(managed between host-device, splitted and assigned to eboy)
        # compile is a hidden to user, that provides common interface to devices
        # context brings magic to user with coordination of eboys, order, and data with compile
        # KOKKOS is highly dependent on the concept of workload (iterations) than data view

        # workshop represents machine, input&output, order, and compile
        workshop = erd.workshop(a, b, "->", c, backend=backend)

        # logical worker entities
        # TODO: how to choose the best configuration per compile
        gofers = erd.gofers(N)

        # generate source code, compile, and run
        gofers.run(workshop)
 
        # do whatever before gofers complete errand

    # When context complets, it ensures that errand is completed

    assert c.sum() == a.sum() + b.sum()
    #assert reduced_c == a.sum() + b.sum()

@pytest.mark.parametrize("backend", test_backends)
def ttest_vecadd2d(backend):

    #NROW, NCOL = 2000, 300
    NROW, NCOL = 20, 3

    a = np.ones((NROW, NCOL))
    b = np.ones((NROW, NCOL))
    c = np.zeros((NROW, NCOL))

    order = os.path.join(here, "res", "vecadd2d.ord")

    with Errand(order, timeout=5) as erd:

        workshop = erd.workshop(a, b, "->", c, backend=backend)

        gofers = erd.gofers(NCOL, NROW)

        gofers.run(workshop)

    assert np.array_equal(c, a+b)


@pytest.mark.parametrize("backend", test_backends)
def ttest_vecadd3d(backend):

    #NROW, NCOL = 2000, 300
    X, Y, Z = 10, 3, 2

    a = np.ones((X, Y, Z))
    b = np.ones((X, Y, Z))
    c = np.zeros((X, Y, Z))

    order = os.path.join(here, "res", "vecadd3d.ord")

    with Errand(order, timeout=5) as erd:

        workshop = erd.workshop(a, b, "->", c, backend=backend)

        gofers = erd.gofers(Z, (X, Y))

        gofers.run(workshop)

    assert np.array_equal(c, a+b)

@pytest.mark.parametrize("backend", test_backends)
def ttest_listadd2d(backend):

    NROW = 2
    NCOL = 3

    a = [[1,1,1], [1,1,1]]
    b = [[1,1,1], [1,1,1]]
    c = [[0,0,0], [0,0,0]]

    order = os.path.join(here, "res", "vecadd2d.ord")

    with Errand(order, timeout=5) as erd:

        workshop = erd.workshop(a, b, "->", c, backend=backend)

        gofers = erd.gofers(NCOL, NROW)

        gofers.run(workshop)

    for i in range(NROW):
        for j in range(NCOL):
            assert c[i][j] == a[i][j] + b[i][j]
