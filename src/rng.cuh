#include <stdbool.h>
#include "ints.h"

///=============================================================================
///                    C implementation of Java Random
///=============================================================================

__device__ static inline void set_seed(u64 *seed, const u64 value)
{
    *seed = (value ^ 0x5deece66dULL) & ((1ULL << 48) - 1);
}

__device__ static inline i32 next(u64 *seed, const i32 bits)
{
    *seed = (*seed * 0x5deece66dULL + 0xb) & ((1ULL << 48) - 1);
    return (i32)((u64)*seed >> (48 - bits));
}

__device__ static inline i32 next_int(u64 *seed, const i32 n)
{
    const i32 m = n - 1;

    if ((m & n) == 0)
    {
        u64 x = n * (u64)next(seed, 31);
        return (i32)((u64)x >> 31);
    }

    i32 bits, val;
    do
    {
        bits = next(seed, 31);
        val = bits % n;
    } while (bits - val + m < 0);

    return val;
}

__device__ static inline bool is_slime(u64 seed, i32 cx, i32 cz)
{
    u64 rnd = seed;
    rnd += (i32)(cx * 0x5ac0db);
    rnd += (i32)(cx * cx * 0x4c1906);
    rnd += (i32)(cz * 0x5f24f);
    rnd += (i32)(cz * cz) * 0x4307a7ULL;
    rnd ^= 0x3ad8025fULL;
    set_seed(&rnd, rnd);
    return next_int(&rnd, 10) == 0;
}
