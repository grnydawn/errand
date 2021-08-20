"""Errand test file"""

import os, numpy as np
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))
NTESTS = 10

N = 100

a = np.ones(N)
b = np.ones(N) * 2
c = np.zeros(N)

order = os.path.join(here, "res", "vecadd1d.ord")
engine = "pthread"

for testid in range(NTESTS):

    with Errand(order, engine=engine, timeout=5) as erd:

        gofers = erd.gofers(N)

        workshop = erd.workshop(a, b, "->", c)

        gofers.run(workshop)

    if c.sum() == a.sum() + b.sum():
        print("%d : SUCCESS" % testid)

    else:
        print("%d : FAILURE %f != %f" % (testid, c.sum(), a.sum() + b.sum()))
