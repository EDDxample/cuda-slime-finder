// v3: each chunk's is_slime is evaluated exactly once (v2 evaluates it ~16x).
//
// Layout: the region is split into tiles of TILE_W output columns x BAND_H
// output rows. A block walks its tile row by row:
//   1. every thread evaluates is_slime for CPT chunks, warp ballots pack the
//      results into 32-chunk bit-words stored in a shared-memory ring buffer,
//   2. each output column extracts its 16-bit window with a funnel shift and
//      updates the vertical running sum: + popcount(newest row window)
//      - popcount(row 16 back), read from the ring.
// The ring holds 32 rows so a single __syncthreads per row is enough: the
// slot being written is never the slot being read (see hazard analysis in the
// row loop), and a power-of-two size makes the slot index a bitwise AND.
//
// Usage:
//   v3 [seed [x0 z0 x_count z_count]]   scan + STATS line
//   v3 --tune                           benchmark all kernel variants
//   v3 --verify                         RNG unit tests + CPU cross-checks

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench.cuh"

static constexpr uint64_t JMULT = 0x5deece66dULL;
static constexpr uint64_t JADD = 0xbULL;
static constexpr uint64_t M48 = (1ULL << 48) - 1ULL;
// The slime seed is XORed with 0x3ad8025f, then Random.setSeed XORs in the
// LCG multiplier; both constants fold into one.
static constexpr uint64_t XMIX = 0x3ad8025fULL ^ 0x5deece66dULL;

static constexpr int32_t BAND_H = 4096; // output rows per tile

// ---------------------------------------------------------------------------
// Seed terms. Matches Java exactly: the x/z polynomials wrap in 32-bit
// arithmetic before being sign-extended into the 64-bit seed sum, so they
// only depend on one axis each and can be precomputed per row/column.
// ---------------------------------------------------------------------------

__host__ __device__ static inline uint64_t se64(int32_t v)
{
    return (uint64_t)(int64_t)v;
}

__host__ __device__ static inline uint64_t make_x_term(int32_t cx)
{
    const uint32_t ux = (uint32_t)cx;
    const int32_t t1 = (int32_t)(ux * 0x5ac0dbu);
    const int32_t t2 = (int32_t)(ux * ux * 0x4c1906u);
    return se64(t1) + se64(t2);
}

__host__ __device__ static inline uint64_t make_z_term(int32_t cz)
{
    const uint32_t uz = (uint32_t)cz;
    const int32_t t1 = (int32_t)(uz * 0x5f24fu);
    const int32_t t2 = (int32_t)(uz * uz);
    return se64(t1) + se64(t2) * 0x4307a7ULL;
}

// ---------------------------------------------------------------------------
// Fast exact is_slime. t = seed + x_term + z_term (mod 2^64).
//
// Bits above 48 in t are garbage but harmless: the low 48 bits of t*JMULT
// only depend on the low 48 bits of t, so the pre-multiply mask is skipped.
// Java's nextInt(10) rejects when bits - bits%10 + 9 overflows int32, which
// simplifies to bits >= 2^31 - 8; in the accepted case the value is 0 iff
// bits is divisible by 10.
// ---------------------------------------------------------------------------
__host__ __device__ static inline bool is_slime_fast(uint64_t t)
{
    uint64_t s = (t ^ XMIX) * JMULT + JADD;
    uint32_t bits = (uint32_t)(s >> 17) & 0x7FFFFFFFu;
    if (bits >= 0x7FFFFFF8u) // probability ~5e-9: replay Java's rejection loop
    {
        s &= M48;
        do
        {
            s = (s * JMULT + JADD) & M48;
            bits = (uint32_t)(s >> 17);
        } while (bits >= 0x7FFFFFF8u);
    }
    // bits % 10 == 0 via the multiply-rotate divisibility test
    // (Hacker's Delight 10-17): 0xCCCCCCCD is 5^-1 mod 2^32, the rotate
    // handles the factor of 2.
    uint32_t q = bits * 0xCCCCCCCDu;
    q = (q >> 1) | (q << 31);
    return q <= 0x19999999u;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// One input row of a tile: evaluate is_slime for CPT chunks per thread, pack
// into bit-words via ballot, then update the vertical sliding sums. HAS_OLD
// and EMIT are compile-time so the warmup rows (r < 15) and the first output
// row (r == 15) carry no data-dependent branches in the steady state.
template <int TPB, int CPT, bool HAS_OLD, bool EMIT, bool FULL>
__device__ __forceinline__ void v3_row(
    uint32_t *rows,
    const uint64_t *__restrict__ z_terms,
    int32_t z_base,
    int32_t r,
    int32_t lane,
    int32_t warp,
    int32_t out_x_base,
    int32_t out_z_base,
    const uint64_t (&xt)[CPT],
    const bool (&active)[CPT],
    const bool (&out_active)[CPT],
    int32_t (&vert)[CPT],
    BestResult &best)
{
    constexpr int WORDS = TPB * CPT / 32;
    constexpr int STRIDE = WORDS + 1;
    constexpr int SLOTS = 32;

    const uint64_t bz = __ldg(&z_terms[z_base + r]); // seed folded in on host
    const int slot = r & (SLOTS - 1);

    uint32_t bal[CPT];
#pragma unroll
    for (int k = 0; k < CPT; ++k)
    {
        const bool p = (FULL || active[k]) && is_slime_fast(bz + xt[k]);
        bal[k] = __ballot_sync(0xffffffffu, p);
        if (lane == 0)
        {
            rows[slot * STRIDE + k * (TPB >> 5) + warp] = bal[k];
        }
    }

    __syncthreads();
    // Hazard check for the single sync: between this row's reads and the next
    // row's writes there is no barrier, but the next row writes slot r+1
    // while this row reads slots r and r-16 (mod SLOTS) - always distinct
    // while SLOTS > 17.

    const uint32_t *new_row = &rows[slot * STRIDE];
    const uint32_t *old_row = &rows[((r + 16) & (SLOTS - 1)) * STRIDE];

#pragma unroll
    for (int k = 0; k < CPT; ++k)
    {
        if (out_active[k])
        {
            // TPB is a multiple of 32, so (tx + k*TPB) & 31 == lane, and the
            // new row's first window word is this warp's own ballot, still in
            // a register.
            const int w = ((warp << 5) + lane + k * TPB) >> 5;
            int32_t v = vert[k];
            v += __popc(__funnelshift_r(bal[k], new_row[w + 1], lane) & 0xFFFFu);
            if (HAS_OLD)
            {
                v -= __popc(__funnelshift_r(old_row[w], old_row[w + 1], lane) & 0xFFFFu);
            }
            vert[k] = v;

            if (EMIT && v > best.score)
            {
                best.score = v;
                best.x = out_x_base + (warp << 5) + lane + k * TPB;
                best.z = out_z_base + (r - 15);
            }
        }
    }
}

template <int TPB, int CPT>
__global__ void __launch_bounds__(TPB) v3_kernel(
    const uint64_t *__restrict__ x_terms,
    const uint64_t *__restrict__ z_terms,
    int32_t x0,
    int32_t z0,
    int32_t x_count,
    int32_t z_count,
    int32_t tiles_x,
    int32_t tiles_z,
    BestResult *__restrict__ block_results)
{
    constexpr int TILE_W = TPB * CPT;
    constexpr int OUT_W = TILE_W - 15;
    constexpr int WORDS = TILE_W / 32;
    constexpr int STRIDE = WORDS + 1; // +1 zero pad word for the funnel shift
    // 17 slots would suffice for the single-sync hazard analysis below, but a
    // power of two turns the slot modulo into a bitwise AND.
    constexpr int SLOTS = 32;

    __shared__ uint32_t rows[SLOTS * STRIDE];
    __shared__ BestResult s_best[TPB];

    const int tx = threadIdx.x;
    const int warp = tx >> 5;
    const int lane = tx & 31;

    // The pad word is never written by ballots; zero it once.
    if (tx < SLOTS)
    {
        rows[tx * STRIDE + WORDS] = 0;
    }

    BestResult best = {-1, 0, 0};
    const int32_t total_tiles = tiles_x * tiles_z;

    for (int32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
    {
        const int32_t x_base = (tile % tiles_x) * OUT_W;
        const int32_t z_base = (tile / tiles_x) * BAND_H;
        const int32_t out_w = min(OUT_W, x_count - x_base);
        const int32_t out_h = min(BAND_H, z_count - z_base);
        const int32_t in_w = out_w + 15;
        const int32_t in_h = out_h + 15;

        // Per-thread x terms are constant for the whole tile.
        uint64_t xt[CPT];
        bool active[CPT];
        bool out_active[CPT];
        int32_t vert[CPT];
#pragma unroll
        for (int k = 0; k < CPT; ++k)
        {
            const int32_t x_local = tx + k * TPB;
            active[k] = x_local < in_w;
            out_active[k] = x_local < out_w;
            xt[k] = x_terms[x_base + (active[k] ? x_local : 0)];
            vert[k] = 0;
        }

        // Covers the pad-word init on the first tile and, on later tiles,
        // keeps this tile's first writes from racing the previous tile's
        // final reads.
        __syncthreads();

        const int32_t out_x_base = x0 + x_base;
        const int32_t out_z_base = z0 + z_base;

        // in_h = out_h + 15 >= 16, so the phase structure below is always
        // valid: 15 warmup rows, one output row without the 16-back subtract,
        // then the steady state. Interior tiles span the full tile width and
        // get a hot path without the edge guard on the slime evaluation.
        if (in_w == TILE_W)
        {
            for (int32_t r = 0; r < 15; ++r)
            {
                v3_row<TPB, CPT, false, false, true>(rows, z_terms, z_base, r, lane, warp,
                                                     out_x_base, out_z_base, xt, active, out_active, vert, best);
            }
            v3_row<TPB, CPT, false, true, true>(rows, z_terms, z_base, 15, lane, warp,
                                                out_x_base, out_z_base, xt, active, out_active, vert, best);
            for (int32_t r = 16; r < in_h; ++r)
            {
                v3_row<TPB, CPT, true, true, true>(rows, z_terms, z_base, r, lane, warp,
                                                   out_x_base, out_z_base, xt, active, out_active, vert, best);
            }
        }
        else
        {
            for (int32_t r = 0; r < 15; ++r)
            {
                v3_row<TPB, CPT, false, false, false>(rows, z_terms, z_base, r, lane, warp,
                                                      out_x_base, out_z_base, xt, active, out_active, vert, best);
            }
            v3_row<TPB, CPT, false, true, false>(rows, z_terms, z_base, 15, lane, warp,
                                                 out_x_base, out_z_base, xt, active, out_active, vert, best);
            for (int32_t r = 16; r < in_h; ++r)
            {
                v3_row<TPB, CPT, true, true, false>(rows, z_terms, z_base, r, lane, warp,
                                                    out_x_base, out_z_base, xt, active, out_active, vert, best);
            }
        }
    }

    s_best[tx] = best;
    __syncthreads();
    for (int stride = TPB / 2; stride > 0; stride >>= 1)
    {
        if (tx < stride && result_better(s_best[tx + stride], s_best[tx]))
        {
            s_best[tx] = s_best[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        block_results[blockIdx.x] = s_best[0];
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

struct DeviceTables
{
    uint64_t *x_terms;
    uint64_t *z_terms;
    BestResult *block_results;
    int32_t max_blocks;
};

static DeviceTables upload_tables(const ScanRegion &r, int32_t max_blocks)
{
    std::vector<uint64_t> hx((size_t)r.x_count + 15);
    std::vector<uint64_t> hz((size_t)r.z_count + 15);
    for (int32_t i = 0; i < r.x_count + 15; ++i)
    {
        hx[(size_t)i] = make_x_term(r.x0 + i);
    }
    for (int32_t i = 0; i < r.z_count + 15; ++i)
    {
        // The seed is folded in here so the kernel adds one term less.
        hz[(size_t)i] = r.seed + make_z_term(r.z0 + i);
    }

    DeviceTables t = {};
    t.max_blocks = max_blocks;
    cuda_check(cudaMalloc((void **)&t.x_terms, hx.size() * sizeof(uint64_t)), "cudaMalloc x_terms");
    cuda_check(cudaMalloc((void **)&t.z_terms, hz.size() * sizeof(uint64_t)), "cudaMalloc z_terms");
    cuda_check(cudaMalloc((void **)&t.block_results, (size_t)max_blocks * sizeof(BestResult)), "cudaMalloc results");
    cuda_check(cudaMemcpy(t.x_terms, hx.data(), hx.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy x_terms");
    cuda_check(cudaMemcpy(t.z_terms, hz.data(), hz.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy z_terms");
    return t;
}

static void free_tables(DeviceTables &t)
{
    cudaFree(t.x_terms);
    cudaFree(t.z_terms);
    cudaFree(t.block_results);
}

template <int TPB, int CPT>
static BestResult run_variant(const ScanRegion &r, const DeviceTables &t, float *kernel_ms)
{
    constexpr int32_t OUT_W = TPB * CPT - 15;
    const int32_t tiles_x = (r.x_count + OUT_W - 1) / OUT_W;
    const int32_t tiles_z = (r.z_count + BAND_H - 1) / BAND_H;
    const int64_t total_tiles = (int64_t)tiles_x * tiles_z;
    const int32_t blocks = (int32_t)(total_tiles < t.max_blocks ? total_tiles : t.max_blocks);

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create");
    cuda_check(cudaEventCreate(&stop), "event create");

    cuda_check(cudaEventRecord(start), "event record");
    v3_kernel<TPB, CPT><<<blocks, TPB>>>(
        t.x_terms, t.z_terms, r.x0, r.z0, r.x_count, r.z_count,
        tiles_x, tiles_z, t.block_results);
    cuda_check(cudaEventRecord(stop), "event record");
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaEventSynchronize(stop), "kernel execution");
    cuda_check(cudaEventElapsedTime(kernel_ms, start, stop), "elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<BestResult> host_results((size_t)blocks);
    cuda_check(cudaMemcpy(host_results.data(), t.block_results, (size_t)blocks * sizeof(BestResult),
                          cudaMemcpyDeviceToHost),
               "copy results");

    BestResult best = {-1, 0, 0};
    for (const BestResult &b : host_results)
    {
        if (result_better(b, best))
        {
            best = b;
        }
    }
    return best;
}

static constexpr int32_t MAX_BLOCKS = 8192;

// Default variant; picked by --tune measurements on the RTX 3060.
static BestResult run_default(const ScanRegion &r, const DeviceTables &t, float *kernel_ms)
{
    return run_variant<128, 8>(r, t, kernel_ms);
}

// ---------------------------------------------------------------------------
// Verification: RNG unit tests + CPU brute-force cross-checks
// ---------------------------------------------------------------------------

// Straight transcription of Java (Random.setSeed + nextInt(10)), used as the
// ground truth.
static bool is_slime_ref(uint64_t seed, int32_t cx, int32_t cz)
{
    uint64_t rnd = seed + make_x_term(cx) + make_z_term(cz);
    rnd ^= 0x3ad8025fULL;
    rnd = (rnd ^ JMULT) & M48;

    while (true)
    {
        rnd = (rnd * JMULT + JADD) & M48;
        const int32_t bits = (int32_t)(rnd >> 17);
        const int32_t val = bits % 10;
        if (bits - val + 9 >= 0)
        {
            return val == 0;
        }
    }
}

static bool is_slime_ref_from_t(uint64_t t)
{
    uint64_t rnd = (t ^ XMIX) & M48;
    while (true)
    {
        rnd = (rnd * JMULT + JADD) & M48;
        const int32_t bits = (int32_t)(rnd >> 17);
        const int32_t val = bits % 10;
        if (bits - val + 9 >= 0)
        {
            return val == 0;
        }
    }
}

static uint64_t xorshift64(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static bool unit_test_rng()
{
    // Random t values.
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 10000000; ++i)
    {
        const uint64_t t = xorshift64(&rng);
        if (is_slime_fast(t) != is_slime_ref_from_t(t))
        {
            printf("FAIL: is_slime_fast mismatch at t=%llu\n", (unsigned long long)t);
            return false;
        }
    }

    // Force the nextInt(10) rejection boundary: build states whose first LCG
    // output lands near 2^31 by inverting the LCG (JMULT is odd, so it is
    // invertible mod 2^48).
    uint64_t inv = 1;
    for (int i = 0; i < 6; ++i) // Newton iteration doubles correct bits
    {
        inv *= 2 - JMULT * inv;
    }
    inv &= M48;

    for (uint32_t bits = 0x7FFFFFE0u; bits <= 0x7FFFFFFFu; ++bits)
    {
        for (int j = 0; j < 64; ++j)
        {
            const uint64_t low17 = xorshift64(&rng) & 0x1FFFFULL;
            const uint64_t s1 = ((uint64_t)bits << 17) | low17;
            const uint64_t state0 = ((s1 - JADD) * inv) & M48;
            const uint64_t t = (state0 ^ XMIX) | (xorshift64(&rng) << 48);
            if (is_slime_fast(t) != is_slime_ref_from_t(t))
            {
                printf("FAIL: rejection-path mismatch at bits=%u t=%llu\n", bits, (unsigned long long)t);
                return false;
            }
        }
    }
    printf("PASS: is_slime_fast matches Java reference (10M random + rejection boundary)\n");
    return true;
}

static bool verify_region(const ScanRegion &r)
{
    // GPU result through the exact production path.
    DeviceTables t = upload_tables(r, MAX_BLOCKS);
    float ms = 0.0f;
    const BestResult gpu = run_default(r, t, &ms);
    free_tables(t);

    // CPU brute force: 2D grid of bits -> column sums -> window sums.
    const int32_t W = r.x_count + 15;
    const int32_t H = r.z_count + 15;
    std::vector<uint8_t> bits((size_t)W * H);
    for (int32_t iz = 0; iz < H; ++iz)
    {
        for (int32_t ix = 0; ix < W; ++ix)
        {
            bits[(size_t)iz * W + ix] = is_slime_ref(r.seed, r.x0 + ix, r.z0 + iz) ? 1 : 0;
        }
    }

    int32_t cpu_best = -1;
    for (int32_t wz = 0; wz < r.z_count; ++wz)
    {
        for (int32_t wx = 0; wx < r.x_count; ++wx)
        {
            int32_t sum = 0;
            for (int32_t dz = 0; dz < 16; ++dz)
            {
                for (int32_t dx = 0; dx < 16; ++dx)
                {
                    sum += bits[(size_t)(wz + dz) * W + (wx + dx)];
                }
            }
            if (sum > cpu_best)
            {
                cpu_best = sum;
            }
        }
    }

    // Recount the GPU-reported window on the CPU grid.
    int32_t gpu_recount = 0;
    const int32_t gx = gpu.x - r.x0;
    const int32_t gz = gpu.z - r.z0;
    if (gx < 0 || gz < 0 || gx >= r.x_count || gz >= r.z_count)
    {
        printf("FAIL: GPU best (%d, %d) outside region\n", gpu.x, gpu.z);
        return false;
    }
    for (int32_t dz = 0; dz < 16; ++dz)
    {
        for (int32_t dx = 0; dx < 16; ++dx)
        {
            gpu_recount += bits[(size_t)(gz + dz) * W + (gx + dx)];
        }
    }

    const bool ok = (gpu.score == cpu_best) && (gpu_recount == gpu.score);
    printf("%s: region (%d, %d) %dx%d -> gpu=%d cpu=%d recount=%d\n",
           ok ? "PASS" : "FAIL", r.x0, r.z0, r.x_count, r.z_count,
           gpu.score, cpu_best, gpu_recount);
    return ok;
}

static int run_verify(uint64_t seed)
{
    bool ok = unit_test_rng();
    const ScanRegion regions[] = {
        {seed, -777, -777, 1500, 1500},    // crosses the origin, partial tiles
        {seed, 123456, -654321, 2000, 300}, // asymmetric, far from origin
        {seed, -1874992, -1874992, 1200, 1200}, // world border corner
        {seed, -1000, 4000, 1017, 4113},   // just past one tile/band boundary
    };
    for (const ScanRegion &r : regions)
    {
        ok = verify_region(r) && ok;
    }
    printf(ok ? "All checks passed.\n" : "VERIFICATION FAILED\n");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------

static int run_tune(const ScanRegion &r)
{
    DeviceTables t = upload_tables(r, MAX_BLOCKS);
    const uint64_t windows = (uint64_t)r.x_count * (uint64_t)r.z_count;

    struct Row
    {
        const char *name;
        BestResult (*fn)(const ScanRegion &, const DeviceTables &, float *);
    };
    const Row rows[] = {
        {"TPB=128 CPT=8 ", run_variant<128, 8>},
        {"TPB=128 CPT=16", run_variant<128, 16>},
        {"TPB=256 CPT=4 ", run_variant<256, 4>},
        {"TPB=256 CPT=8 ", run_variant<256, 8>},
        {"TPB=256 CPT=16", run_variant<256, 16>},
        {"TPB=512 CPT=4 ", run_variant<512, 4>},
        {"TPB=512 CPT=8 ", run_variant<512, 8>},
    };

    printf("Tuning on %d x %d windows...\n", r.x_count, r.z_count);
    for (const Row &row : rows)
    {
        float ms = 0.0f;
        const BestResult best = row.fn(r, t, &ms); // warmup
        float ms2 = 0.0f;
        row.fn(r, t, &ms2);
        ms = ms < ms2 ? ms : ms2;
        printf("  %s: %8.2f ms  %.3e windows/s  (best=%d)\n",
               row.name, ms, (double)windows / ((double)ms * 1e-3), best.score);
    }
    free_tables(t);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--verify") == 0)
    {
        return run_verify(DEFAULT_REGION.seed);
    }
    if (argc >= 2 && strcmp(argv[1], "--tune") == 0)
    {
        return run_tune(DEFAULT_REGION);
    }

    const ScanRegion region = parse_region(argc, argv);
    const uint64_t windows = (uint64_t)region.x_count * (uint64_t)region.z_count;

    printf("v3: scanning %d x %d windows from (%d, %d), seed %llu\n",
           region.x_count, region.z_count, region.x0, region.z0,
           (unsigned long long)region.seed);

    DeviceTables tables = upload_tables(region, MAX_BLOCKS);
    float kernel_ms = 0.0f;
    const BestResult best = run_default(region, tables, &kernel_ms);
    free_tables(tables);

    print_best(best);
    print_stats(windows, kernel_ms);
    return 0;
}
