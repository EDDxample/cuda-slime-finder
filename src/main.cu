#include <stdio.h>
#include "ints.h"
#include "queue.cuh"
#include "rng.cuh"

#define CHUNK_WORLD_BORDER 30000000 / 16 - 8

__device__ i16 compute_column(u64 seed, i32 xdx, i32 z)
{
    i16 sum = 0;
    for (i32 dz = -8; dz < 8; ++dz)
    {
        sum += is_slime(seed, xdx, z + dz);
    }
    return sum;
}

__device__ void iter_row(u64 seed, i32 thread_id, i32 z, i16 *out_values, i32 *out_x)
{
    i16 best = 0;
    i32 best_x = 0;

    i16 slime_count = -1;
    Queue_t queue = {0, 0, 0, {0}};

    for (i32 x = -CHUNK_WORLD_BORDER; x < CHUNK_WORLD_BORDER; ++x)
    {
        if (slime_count == -1)
        {
            slime_count = 0;
            for (i32 dx = -8; dx < 8; ++dx)
            {
                i16 column = compute_column(seed, x + dx, z);
                slime_count += column;
                queue_write(&queue, column);
            }
            continue;
        }

        slime_count -= queue_read(&queue);

        i16 new_column = compute_column(seed, x + 7, z);
        slime_count += new_column;
        queue_write(&queue, new_column);

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

        // get the closest point with the highest value
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

__global__ void slime_finder_kernel(u64 seed, i16 *out_values, i32 *out_x)
{
    i32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    i32 thread_count = blockDim.x * gridDim.x;
    i32 z = thread_id - thread_count / 2;

    iter_row(seed, thread_id, z, out_values, out_x);
}

int main()
{
    u64 seed = 8354031675596398786ULL;

    // kernel parameters (GPU based)
    i32 blocks_per_grid = 32 * 2;
    i32 threads_per_block = 1024;
    i32 thread_count = blocks_per_grid * threads_per_block;

    i32 out_values_len = sizeof(i16) * thread_count;
    i32 out_x_len = sizeof(i32) * thread_count;

    // create host output arrays
    i16 *host_values = (i16 *)malloc(out_values_len);
    i32 *host_x = (i32 *)malloc(out_x_len);

    // create device output arrays
    i16 *device_values;
    i32 *device_x;
    cudaMalloc((void **)&device_values, out_values_len);
    cudaMalloc((void **)&device_x, out_x_len);

    // launch kernel
    slime_finder_kernel<<<blocks_per_grid, threads_per_block>>>(seed, device_values, device_x);

    // copy the result back to the host
    cudaMemcpy(host_values, device_values, out_values_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x, device_x, out_x_len, cudaMemcpyDeviceToHost);

    print_best_chunk(host_values, host_x, thread_count);

    // cleanup
    cudaFree(device_values);
    cudaFree(device_x);
    free(host_values);
    free(host_x);
    return 0;
}
