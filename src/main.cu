#include <stdio.h>
#include "rng.cuh"
#include "ints.h"

__device__ i16 compute_column(u64 seed, i32 xdx, i32 z)
{
    i16 sum = 0;
    for (i32 dz = -8; dz < 8; ++dz)
    {
        sum += is_slime(seed, xdx, z + dz);
    }
    return sum;
}

__device__ i16 compute_center(u64 seed, i32 x, i32 z)
{
    i16 sum = 0;
    for (i32 dx = -8; dx < 8; ++dx)
    {
        sum += compute_column(seed, x + dx, z);
    }
    return sum;
}

__device__ void iter_row(u64 seed, i32 thread_id, i32 z, i16 *out_values, i32 *out_x)
{
    i16 best = 0;
    i32 best_x = 0;

    i16 slime_count = -1;
    for (i32 x = -1875000; x < 1875000; ++x)
    {
        if (slime_count == -1)
        {
            slime_count = compute_center(seed, x, z);
            continue;
        }
        slime_count -= compute_column(seed, x - 9, z);
        slime_count += compute_column(seed, x + 7, z);
        if (slime_count > best)
        {
            best = slime_count;
            best_x = x;
        }
    }

    out_values[thread_id] = best;
    out_x[thread_id] = best_x;
}

__host__ void print_best_chunk(i16 *out_values, i32 *out_x, i32 thread_count)
{
    printf("\nComputing best row...\n");
    i16 best = 0;
    i32 best_x = 0;
    i32 best_z = 0;

    for (i32 thread_id = 0; thread_id < thread_count; ++thread_id)
    {
        i32 value = out_values[thread_id];
        i32 x = out_x[thread_id];
        i32 z = thread_id - thread_count / 2;

        if (value > best || (value == best && abs(x + z) < abs(best_x + best_z)))
        {
            best = value;
            best_x = x;
            best_z = z;
            printf("Intermediate: %d (%d, %d)\n", best, best_x, best_z);
        }
    }

    printf("Result: %d (%d, %d) (%d, %d)\n", best, best_x, best_z, best_x * 16 + 8, best_z * 16 + 8);
}

__global__ void slime_finder_kernel(u64 seed, i16 *out_d, i32 *outx_d)
{
    i32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    i32 thread_count = blockDim.x * gridDim.x;
    i32 z = thread_id - thread_count / 2;

    iter_row(seed, thread_id, z, out_d, outx_d);
}

int main()
{
    u64 seed = 8354031675596398786ULL;

    // kernel parameters (GPU based)
    i32 blocks_per_grid = 32 * 2;
    i32 threads_per_block = 1024;
    i32 thread_count = blocks_per_grid * threads_per_block;

    i32 out_len = sizeof(i16) * thread_count;
    i32 outx_len = sizeof(i32) * thread_count;

    // create host output arrays
    i16 *out_h = (i16 *)malloc(out_len);
    i32 *outx_h = (i32 *)malloc(outx_len);

    // create device output arrays
    i16 *out_d;
    i32 *outx_d;
    cudaMalloc((void **)&out_d, out_len);
    cudaMalloc((void **)&outx_d, outx_len);

    // launch kernel
    slime_finder_kernel<<<blocks_per_grid, threads_per_block>>>(seed, out_d, outx_d);

    // copy the result back to the host
    cudaMemcpy(out_h, out_d, out_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(outx_h, outx_d, outx_len, cudaMemcpyDeviceToHost);

    print_best_chunk(out_h, outx_h, thread_count);

    // cleanup
    cudaFree(out_d);
    cudaFree(outx_d);
    free(out_h);
    free(outx_h);
    return 0;
}
