
CC := nvcc
SH_OPTS := --compiler-options '-fPIC' --shared

cuwrap: cuwrap.cu
	${CC} ${SH_OPTS} -o $@.so $^ 

hello.exe: cuhello.cu cuwrap
	#${CC} -o $@ --compiler-options "-Wl,-rpath,${PWD}" cuwrap.so $<
	${CC} -o $@ --linker-options "-rpath,${PWD}" cuwrap.so $<

clean:
	rm -f *.o *.so *.exe
