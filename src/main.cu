#include <inttypes.h>
#include <stdio.h>
#include "rng.cuh"
#include "queue.cuh"

__constant__ uint64_t SEED;

__device__ uint8_t find_best_x(int thread_id, int z, uint8_t *out_d, int32_t *out_dx)
{
    struct Queue q = {0, 0, 0, {}};

    int slime_count = -1;
    int best = 0;
    int best_x = 0;

    for (int x = -3750000; x < 3750000; ++x)
    {
        // compute full slime count if needed
        if (slime_count == -1)
        {
            slime_count = 0;
            for (int dx = -8; dx < 8; ++dx)
            {
                uint8_t row_count = 0;
                for (int dz = -8; dz < 8; ++dz)
                {
                    int flag = is_slime(SEED, x + dx, z + dz);
                    row_count += flag;
                }
                slime_count += row_count;

                // add computed row to queue
                queue_write(&q, row_count);
            }
            best = slime_count;
            best_x = x;
            continue;
        }

        // fetch prev row
        uint8_t prev_row = 0;
        queue_read(&q, &prev_row);

        slime_count -= prev_row;

        // compute next row
        uint8_t next_row = 0;
        for (int dz = -8; dz < 8; ++dz)
        {
            next_row += is_slime(SEED, x + 8, z + dz);
        }
        slime_count += next_row;
        queue_write(&q, next_row);

        if (slime_count > best)
        {
            best = slime_count;
            best_x = x;
        }
    }

    out_d[thread_id] = best;
    out_dx[thread_id] = best_x;
}

__global__ void slime_finder_kernel(uint8_t *out_d, int32_t *out_dx)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;
    int z = thread_id - size / 2;

    find_best_x(thread_id, z, out_d, out_dx);
}

__host__ void print_best_row(uint8_t *out_h, int32_t *out_x, int out_l)
{
    printf("computing best row...\n");
    uint8_t best = 0;
    uint8_t best_x = 0;
    uint8_t best_z = 0;

    for (int z = 0; z < out_l; ++z)
    {
        if (out_h[z] > best)
        {
            best = out_h[z];
            best_x = out_x[z];
            best_z = z;
            printf("Intermediate: %d (%d, %d)\n", best, best_x, best_z);
        }
    }

    printf("Result: %d (%d, %d)\n", best, best_x, best_z);
}

int main(int argc, char const *argv[])
{
    // if (argc <= 1)
    // {
    //     printf("Usage: slime_finder <seed>");
    //     return 1;
    // }
    // uint64_t seed = atoll(argv[1]);

    // load seed into gpu
    uint64_t seed = 8354031675596398786ULL;
    cudaMemcpyToSymbol(SEED, &seed, sizeof(uint64_t));

    // kernel parameters
    int blocks_per_grid = 32 * 2;
    int threads_per_block = 1024;
    int out_l = sizeof(uint8_t) * blocks_per_grid * threads_per_block;

    // create host output array
    uint8_t *out_h = (uint8_t *)malloc(out_l);
    int32_t *out_hx = (int32_t *)malloc(out_l);

    // create device output array
    uint8_t *out_d;
    int32_t *out_dx;
    cudaMalloc((void **)&out_d, out_l);
    cudaMalloc((void **)&out_dx, out_l * 4);

    // launch kernel
    slime_finder_kernel<<<blocks_per_grid, threads_per_block>>>(out_d, out_dx);

    // copy the result back to the host
    cudaMemcpy(out_h, out_d, out_l, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_hx, out_dx, out_l, cudaMemcpyDeviceToHost);

    print_best_row(out_h, out_hx, out_l);

    // cleanup
    cudaFree(out_d);
    cudaFree(out_dx);
    free(out_h);
    free(out_hx);
    return 0;
}
