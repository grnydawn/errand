import numpy as np
from errand import Errand

N1 = 10
N2 = 20

a = np.ones((N1, N2))
b = np.ones((N1, N2))
c = np.zeros((N1, N2))

#with Errand("order.ord") as erd:
with Errand("vecadd2d.ord") as erd:

    # call N1 teams of N2 gofers 
    gofers = erd.gofers(N1, N2)

    # build workshop with input and output, where actual work takes place
    workshop = erd.workshop(a, b, "->", c)

    # let gofers do their work
    gofers.run(workshop)

    # do your work while gofers are doing their work

# check the result when the errand is completed
if np.array_equal(c, a+b):
    print("SUCCESS!")

else:
    print("FAILURE!")
