import numpy as np
from errand import Errand

N = 100

a = np.ones((N, N))
b = np.ones((N, N)) * 2
c = np.zeros((N, N))

with Errand("vecadd2d.ord") as erd:

    # call gofers
    gofers = erd.gofers(N, N)

    # build workshop with input and output, where actual work takes place
    workshop = erd.workshop(a, b, "->", c)

    # let gofers do their work
    gofers.run(workshop)

# check result
assert np.array_equal(c, a+b)
