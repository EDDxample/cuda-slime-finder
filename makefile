build:
	nvcc src/main.cu -o build/slime_finder

clean:
	-rm -rf build/*