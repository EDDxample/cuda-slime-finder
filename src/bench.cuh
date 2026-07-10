// Shared helpers for benchmarkable slime finder binaries.
//
// Common CLI:   binary [seed [x0 z0 x_count z_count]]
//   (x0, z0)          top-left chunk coord of the first 16x16 window
//   x_count, z_count  number of window positions to scan in each axis
//
// Every binary prints a machine-readable line:
//   STATS windows=<N> kernel_ms=<ms> rate=<windows per second>
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct ScanRegion
{
    uint64_t seed;
    int32_t x0;
    int32_t z0;
    int32_t x_count;
    int32_t z_count;
};

// Default benchmark region: 2^20 x 2^16 windows centered on the origin.
static const ScanRegion DEFAULT_REGION = {
    8354031675596398786ULL,
    -524288,
    -32768,
    1048576,
    65536,
};

static inline ScanRegion parse_region(int argc, char **argv)
{
    ScanRegion r = DEFAULT_REGION;
    if (argc >= 2)
    {
        r.seed = strtoull(argv[1], nullptr, 10);
    }
    if (argc >= 6)
    {
        r.x0 = (int32_t)strtol(argv[2], nullptr, 10);
        r.z0 = (int32_t)strtol(argv[3], nullptr, 10);
        r.x_count = (int32_t)strtol(argv[4], nullptr, 10);
        r.z_count = (int32_t)strtol(argv[5], nullptr, 10);
    }
    return r;
}

static inline void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

static inline void print_stats(uint64_t windows, float kernel_ms)
{
    printf("STATS windows=%llu kernel_ms=%.3f rate=%.3e\n",
           (unsigned long long)windows, kernel_ms, (double)windows / ((double)kernel_ms * 1e-3));
    printf("Duration: %.3f s\n", (double)kernel_ms * 1e-3);
}

struct BestResult
{
    int32_t score;
    int32_t x; // top-left chunk x of the 16x16 window
    int32_t z; // top-left chunk z of the 16x16 window
};

__host__ __device__ static inline int32_t iabs32(int32_t v)
{
    return v < 0 ? -v : v;
}

// Higher score wins; ties broken by distance to the origin.
__host__ __device__ static inline bool result_better(const BestResult &a, const BestResult &b)
{
    if (a.score != b.score)
    {
        return a.score > b.score;
    }
    const uint32_t ad = (uint32_t)iabs32(a.x) + (uint32_t)iabs32(a.z);
    const uint32_t bd = (uint32_t)iabs32(b.x) + (uint32_t)iabs32(b.z);
    if (ad != bd)
    {
        return ad < bd;
    }
    if (a.x != b.x)
    {
        return a.x < b.x;
    }
    return a.z < b.z;
}

static inline void print_best(const BestResult &best)
{
    printf("Best 16x16 window: %d slime chunks\n", best.score);
    printf("  chunk top-left: (%d, %d)   chunk center: (%d, %d)\n",
           best.x, best.z, best.x + 8, best.z + 8);
    printf("  block center:   (%d, %d)\n", best.x * 16 + 128, best.z * 16 + 128);
}
