NVCC = nvcc
FLAGS = -O3 -arch=sm_86

.PHONY: build v2 v3 verify tune bench clean

build: v2 v3

v2:
	$(NVCC) $(FLAGS) src/v2.cu -o build/v2

v3:
	$(NVCC) $(FLAGS) src/v3.cu -o build/v3

# CPU brute-force cross-checks + RNG unit tests
verify: v3
	./build/v3 --verify

# benchmark every kernel variant (block size x chunks per thread)
tune: v3
	./build/v3 --tune

# record the current build's throughput and refresh bench/progress.png
bench: v3
	python bench/bench.py run build/v3.exe --label v3

clean:
	-rm -rf build/*
