===============
Getting started
===============

**errand** makes use of conventional programming tools that you may be already familar with. For example, **errand** uses Nvidia CUDA compiler or AMD HIP compiler if needed. **errand** takes responsibilities of data movements between GPU and CPU so that you can focus on computation in CUDA, HIP, OpenAcc(C++), or Pthread(C++).

Installation
-------------

The easiest way to install errand is to use the pip python package manager. 

        >>> pip install errand

You can install errand from github code repository if you want to try the latest version.

        >>> git clone https://github.com/grnydawn/errand.git
        >>> cd errand
        >>> python setup.py install


NumPy array example in CUDA(Nvidia) or HIP(AMD)
-------------------------------------------------------

To run the example, create two source files in a folder as shown below, and run the Python script as usual.
The example assumes that at least one of the following compilers is usable: CUDA (nvcc), HIP(hipcc), C++ OpenAcc(GNU >=10), and Pthread C++ compiler(GNU)).

::

	>>> python main.py

The following Python code demonstrates how to compute numpy arrays using multiple programming frameworks including Cuda, Hip, OpenAcc(GNU), or Pthread(GNU). Errand automatically checks and uses one of frameworks available on the system.

Python code (main.py)

::

	# This example shows how to add numpy arrays using Errand.

	import numpy as np
	from errand import Errand

	N1 = 10
	N2 = 20

	a = np.ones((N1, N2))
	b = np.ones((N1, N2))
	c = np.zeros((N1, N2))

	# creates an errand context with an "order"
	with Errand("order.ord") as erd:

		# build workshop with input(a, b) and output(c)
		workshop = erd.workshop(a, b, "->", c)

		# call N1 teams of N2 gofers 
		gofers = erd.gofers(N1, N2)

		# let gofers do their work at the workshop
		gofers.run(workshop)

		# do your work below while gofers are doing their work

	# check the result when the errand is completed
	if np.array_equal(c, a+b):
		print("SUCCESS!")

	else:
		print("FAILURE!")


Errand takes in charge of data movements and thread generation. User is responsible for specifying computation in an order file. The order file below defines an element-wise addition of 2 dimensional array in multiple programming framework including Cuda, Hip, OpenAcc-C++, and PThread.

For convinience, Errand provides user with a Numpy ndarray-like interface to the input and output arguments as demonstrated below. For example, an array can be accessed through indices and the shape array is informed with shape member method.

Order code (order.ord)

::

	[cuda, hip]

		// N1 teams are interpreted to Cuda/Hip blocks
		// N2 gofers of a team are interpreted to Cuda/Hip threads

		int row = blockIdx.x;
		int col = threadIdx.x;

		// the input and output variables keep the convinience of numpy

		if (row < x.shape(0) && col < x.shape(1))
			c(row, col) = a(row, col) + b(row, col);

	[openacc-c++]

		#pragma acc loop gang
		for (int row = 0; row < a.shape(0); row++) {

			#pragma acc loop vector
			for (int col = 0; col < a.shape(1); col++) {
				c(row, col) = a(row, col) + b(row, col);
			}
		}

	[pthread]

		int row = a.unravel_index(ERRAND_GOFER_ID, 0);
		int col = a.unravel_index(ERRAND_GOFER_ID, 1);

		if (row < a.shape(0) && col < a.shape(1) )
			c(row, col) = a(row, col) + b(row, col);

