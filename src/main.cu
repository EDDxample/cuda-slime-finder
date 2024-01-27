#include <inttypes.h>
#include <stdio.h>
#include "rng.cuh"

__device__ int16_t compute_column(uint64_t seed, int32_t xdx, int32_t z)
{
    int16_t sum = 0;
    for (int32_t dz = -8; dz < 8; ++dz)
    {
        sum += is_slime(seed, xdx, z + dz);
    }
    return sum;
}

__device__ int16_t compute_center(uint64_t seed, int32_t x, int32_t z)
{
    int16_t sum = 0;
    for (int32_t dx = -8; dx < 8; ++dx)
    {
        sum += compute_column(seed, x + dx, z);
    }
    return sum;
}

__device__ void iter_row(uint64_t seed, int32_t thread_id, int32_t z, int16_t *out_values, int32_t *out_x)
{
    int best = 0;
    int best_x = 0;

    int slime_count = -1;
    for (int x = -1875000; x < 1875000; ++x)
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

__host__ void print_best_chunk(int16_t *out_values, int32_t *out_x, int32_t thread_count)
{
    printf("\nComputing best row...\n");
    int16_t best = 0;
    int32_t best_x = 0;
    int32_t best_z = 0;

    for (int thread_id = 0; thread_id < thread_count; ++thread_id)
    {
        int value = out_values[thread_id];
        int x = out_x[thread_id];
        int z = thread_id - thread_count / 2;

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

__global__ void slime_finder_kernel(uint64_t seed, int16_t *out_d, int32_t *outx_d)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * gridDim.x;
    int z = thread_id - thread_count / 2;

    iter_row(seed, thread_id, z, out_d, outx_d);
}

int main(int argc, char const *argv[])
{
    uint64_t seed = 8354031675596398786ULL;

    // kernel parameters (GPU based)
    int blocks_per_grid = 32 * 2;
    int threads_per_block = 1024;
    int thread_count = blocks_per_grid * threads_per_block;

    int out_len = sizeof(int16_t) * thread_count;
    int outx_len = sizeof(int32_t) * thread_count;

    // create host output arrays
    int16_t *out_h = (int16_t *)malloc(out_len);
    int32_t *outx_h = (int32_t *)malloc(outx_len);

    // create device output arrays
    int16_t *out_d;
    int32_t *outx_d;
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
