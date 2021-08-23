===============
Errand Basics
===============

Writing an **errand** has two parts. The first part is to define errand in Python code. More specifically, user defines a workshop(analogous to h/w such as GPU) and gofers(analogous to threads). The second part is to define an order that the gofers run in the workshop.

Writing a workshop and gofers in Python code
===============================================

Following code shows typical way to define a workshop and gofers.

::

	with Errand("order.ord") as erd:

		workshop = erd.workshop(a, b, "->", c)

		gofers = erd.gofers(NUM_TEAMS, NUM_GOFERS_PER_TEAM)

		gofers.run(workshop)

		# do your work below while gofers are doing their work

	# The output from the errand is available after the end of the Errand context.


Errand uses Python context manager to wrap all activities under "errand". In the first line an Errand object is created with a file path to "order" file.

Workshop is created through Errand context handler. Errand accepts numpy ndarray or Python objects that can be converted to ndarray. Arguments is splited by an arrow argument, "->". Left arguments of the arrow argument are input arguments, and right arguments are output arguments. In the example, "a" and "b" variables are inputs and "c" is an output variable.

Next, gofers runs the order on the workshop, where technically Errand generates source code based on "order.ord" and compiles it to an shared library, and load & run the shared library.

Errand runs the "errand" asynchronously. Therefore, Errand natually supports overlapping computations between host machine and target machine.

Errand collect output from target machine and populate it to output variable.


Writing an order in various programming frameworks
===================================================

As of this writing, Errand supports Cuda, Hip, OpenAcc(GNU C++), and PThread(GNU C++). More programming frameworks and compilers will be supported.

Order file is composed of one implicit Python section and multiple programming framework sections.

"order.ord"

::

	# implicit Python section

	enable_pthread = False # control if pthread section is enabled.

	[cuda, hip]

		// first section is for cuda as well as hip

	[openacc-c++: -O3]

		// second section for openacc-c++

	[pthread@enable=enable_pthread]

		// third section for pthread


From the beginning of the file to the right before of the first named section is the implicit Python section.  You can put any Python code that you can write in a Python script file. The code will be executed when the order file is loaded and the variables created will be used to control the following sections. For example, "enable_pthread" is set to False and that disables the third pthread section from execution candidates. You can create multiple versions of the same section and activates only one of them depending on system, compiler, test cases, and so on

The first named section has two target frameworks, "cuda" and "hip". In many cases, it is possible that a code can be compiled by both compilers as they share many features in common. Incompatibilities between cuda and hip in programming perspective exist on mainly data movement between CPU and GPU, and Errand takes care of the differnce behind.

User can control many aspects of compilation. In the example for openacc-c++, user added -O3 compiler options.

For the purpose of explanation, the last pthread section is disabled as explaned in the first implicit Python section.


Numpy inputs to Errand
===================================================

Errand expects input argument to Errand workshop are Numpy ndarray or Python objects that can be converted to ndarray. The ndarray provides various information that Errand pulls from and populates to the Errand genrated source file.

::

	import numpy as np
	from errand import Errand

	N1 = 10
	N2 = 20

	a = np.ones((N1, N2))
	b = np.ones((N1, N2))
	c = np.zeros((N1, N2))


	with Errand("order.ord") as erd:

		...


		workshop = erd.workshop(a, b, "->", c)

In the example, "a", "b" are input ndarray variables and "c" is output ndarray variable. "->" is a visual sign that seperates aruments between inputs and outputs.

You can use Python list instead of ndarray. However, the list is converted into ndarray Errand interanlly.

::

	NROW = 2
	NCOL = 3

	a = [[1,1,1], [1,1,1]]
	b = [[1,1,1], [1,1,1]]
	c = [[0,0,0], [0,0,0]]

	with Errand("order.ord") as erd:

		workshop = erd.workshop(a, b, "->", c)

		...
