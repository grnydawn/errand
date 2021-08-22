import os
import numpy as np
from errand import Errand

here = os.path.dirname(os.path.abspath(__file__))

N1 = 10
N2 = 20

a = np.ones((N1, N2))
b = np.ones((N1, N2))
c = np.zeros((N1, N2))

#engine = "cuda"
#engine = "hip"
#engine = "pthread"
engine = "openacc-c++"

#with Errand(os.path.join(here, "vecadd2d.ord")) as erd:
with Errand(os.path.join(here, "vecadd2d.ord"), engine=engine) as erd:

    # build workshop with input and output, where actual work takes place
    workshop = erd.workshop(a, b, "->", c)

    # call N1 teams of N2 gofers 
    gofers = erd.gofers(N1, N2)

    # let gofers do their work
    gofers.run(workshop)

    # do your work while gofers are doing their work

# check the result when the errand is completed
if np.array_equal(c, a+b):
    print("SUCCESS!")

else:
    print("FAILURE!")
