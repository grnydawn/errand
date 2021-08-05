import numpy as np
from errand import Errand

N = 100

a = np.ones(N)
b = np.ones(N) * 2
c = np.zeros(N)

with Errand("vecadd.ord") as erd:

        # call gofers
        gofers = erd.gofers(N)

        # build workshop with input and output, where actual work takes place
        workshop = erd.workshop(a, b, "->", c)

        # let gofers do their work
        gofers.run(workshop)

# check result
assert c.sum() == a.sum() + b.sum()
