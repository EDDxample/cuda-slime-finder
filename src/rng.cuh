#include <stdbool.h>
#include <stdint.h>

///=============================================================================
///                    C implementation of Java Random
///=============================================================================

__device__ static inline void set_seed(uint64_t *seed, const uint64_t value)
{
    *seed = (value ^ 0x5deece66dULL) & ((1ULL << 48) - 1);
}

__device__ static inline int32_t next(uint64_t *seed, const int32_t bits)
{
    *seed = (*seed * 0x5deece66dULL + 0xb) & ((1ULL << 48) - 1);
    return (int32_t)((uint64_t)*seed >> (48 - bits));
}

__device__ static inline int32_t next_int(uint64_t *seed, const int32_t n)
{
    const int32_t m = n - 1;

    if ((m & n) == 0)
    {
        uint64_t x = n * (uint64_t)next(seed, 31);
        return (int32_t)((uint64_t)x >> 31);
    }

    int32_t bits, val;
    do
    {
        bits = next(seed, 31);
        val = bits % n;
    } while (bits - val + m < 0);

    return val;
}

__device__ static inline bool is_slime(uint64_t seed, int32_t cx, int32_t cz)
{
    uint64_t rnd = seed;
    rnd += (int32_t)(cx * 0x5ac0db);
    rnd += (int32_t)(cx * cx * 0x4c1906);
    rnd += (int32_t)(cz * 0x5f24f);
    rnd += (int32_t)(cz * cz) * 0x4307a7ULL;
    rnd ^= 0x3ad8025fULL;
    set_seed(&rnd, rnd);
    return next_int(&rnd, 10) == 0;
}
