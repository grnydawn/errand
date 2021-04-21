"""Errand test file"""

import os, numpy as np
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))

def test_cuda():

    N = 10

    a = np.ones(N)
    b = np.ones(N)
    c = np.zeros(N)

    esf = os.path.join(here, "res", "vecadd.esf")

    with Errand(esf, engine="cuda") as erd:

        erd.memcpy2device(a)
        erd.memcpy2device(b)

        #erd.run[N](a, b, N)
        erd.run(a, b, N)

        erd.memcpy2host(c)

    assert c.sum() == a.sum() + b.sum()
