# Slime Finder
A Slime Chunk finder implementation in CUDA to find the 16x16 area with the most slime chunks around.

## Commands
```sh
# compile
nvcc src/main.cu -o build/slime_finder

# run
./build/slime_finder
```

## Algorithms
- v1: 
    - Each device thread computes 1 full row of chunks at a given Z coord.
    - The host finds the best row from the results of the previous step.
    - Duration: ~1:50 secs.
- v2:
    - Same algorithm as v1.
    - A queue is used to cache previous computed columns.
    - Duration: ~0:45 secs.
- v3 (TODO):
    - Takaoka's parallel algorithm (check sources).

## Sources
- Cubiome's [implementation of java random in C](https://github.com/Cubitect/cubiomes/blob/master/rng.h).
- Tadao Takaoka's paper on [Efficient Parallel Algorithms for the Maximum Subarray Problem](https://crpit.scem.westernsydney.edu.au/confpapers/CRPITV152Takaoka.pdf).
