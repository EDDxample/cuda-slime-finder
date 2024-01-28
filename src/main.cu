#include <stdio.h>
#include <stdint.h>

#include "queue.cuh"
#include "rng.cuh"

#define CHUNK_WORLD_BORDER 30000000 / 16 - 8

__device__ int16_t compute_column(uint64_t seed, int32_t xdx, int32_t z)
{
    int16_t sum = 0;
    for (int32_t dz = -8; dz < 8; ++dz)
    {
        sum += is_slime(seed, xdx, z + dz);
    }
    return sum;
}

__device__ void iter_row(uint64_t seed, int32_t thread_id, int32_t z, int16_t *out_values, int32_t *out_x)
{
    int16_t best = 0;
    int32_t best_x = 0;

    int16_t slime_count = -1;
    Queue_t queue = {0, 0, 0, {0}};

    for (int32_t x = -CHUNK_WORLD_BORDER; x < CHUNK_WORLD_BORDER; ++x)
    {
        if (slime_count == -1)
        {
            slime_count = 0;
            for (int32_t dx = -8; dx < 8; ++dx)
            {
                int16_t column = compute_column(seed, x + dx, z);
                slime_count += column;
                queue_write(&queue, column);
            }
            continue;
        }

        slime_count -= queue_read(&queue);

        int16_t new_column = compute_column(seed, x + 7, z);
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

__host__ void print_best_chunk(int16_t *out_values, int32_t *out_x, int32_t thread_count)
{
    printf("\nComputing best row...\n");
    int16_t best = 0;
    int32_t best_x = 0;
    int32_t best_z = 0;

    for (int32_t thread_id = 0; thread_id < thread_count; ++thread_id)
    {
        int32_t value = out_values[thread_id];
        int32_t x = out_x[thread_id];
        int32_t z = thread_id - thread_count / 2;

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

__global__ void slime_finder_kernel(uint64_t seed, int16_t *out_values, int32_t *out_x)
{
    int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t thread_count = blockDim.x * gridDim.x;
    int32_t z = thread_id - thread_count / 2;

    iter_row(seed, thread_id, z, out_values, out_x);
}

int main()
{
    uint64_t seed = 8354031675596398786ULL;

    // kernel parameters (GPU based)
    int32_t blocks_per_grid = 32 * 2;
    int32_t threads_per_block = 1024;
    int32_t thread_count = blocks_per_grid * threads_per_block;

    int32_t out_values_len = sizeof(int16_t) * thread_count;
    int32_t out_x_len = sizeof(int32_t) * thread_count;

    // create host output arrays
    int16_t *host_values = (int16_t *)malloc(out_values_len);
    int32_t *host_x = (int32_t *)malloc(out_x_len);

    // create device output arrays
    int16_t *device_values;
    int32_t *device_x;
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
