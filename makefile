build_cuda:
	nvcc src/main.cu -o build/slime_finder

build_c:
	gcc src/main.c -o build/c
