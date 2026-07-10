// v2 baseline, parametrized for benchmarking.
// One thread per output row; a queue caches the 16-tall column sums while the
// window slides along x. This is the original main.cu algorithm (see git
// history) wrapped in the shared benchmark harness.

#include <stdint.h>
#include <stdio.h>

#include "bench.cuh"
#include "queue.cuh"
#include "rng.cuh"

__device__ static int16_t compute_column(uint64_t seed, int32_t x, int32_t z_center)
{
    int16_t sum = 0;
    for (int32_t dz = -8; dz < 8; ++dz)
    {
        sum += is_slime(seed, x, z_center + dz);
    }
    return sum;
}

__global__ static void slime_finder_kernel(ScanRegion region, int16_t *out_values, int32_t *out_x)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= region.z_count)
    {
        return;
    }

    // Window with top-left (x, z) has center coords (x + 8, z + 8): the
    // original code scans centers with the window spanning [c - 8, c + 8).
    const int32_t z_center = region.z0 + 8 + thread_id;
    const int32_t x_center_first = region.x0 + 8;
    const int32_t x_center_last = x_center_first + region.x_count; // exclusive

    int16_t best = -1;
    int32_t best_x = 0;

    int16_t slime_count = -1;
    Queue_t queue = {0, 0, 0, {0}};

    for (int32_t x = x_center_first; x < x_center_last; ++x)
    {
        if (slime_count == -1)
        {
            slime_count = 0;
            for (int32_t dx = -8; dx < 8; ++dx)
            {
                int16_t column = compute_column(region.seed, x + dx, z_center);
                slime_count += column;
                queue_write(&queue, column);
            }
        }
        else
        {
            slime_count -= queue_read(&queue);
            int16_t new_column = compute_column(region.seed, x + 7, z_center);
            slime_count += new_column;
            queue_write(&queue, new_column);
        }

        if (slime_count > best)
        {
            best = slime_count;
            best_x = x - 8; // report top-left
        }
    }

    out_values[thread_id] = best;
    out_x[thread_id] = best_x;
}

int main(int argc, char **argv)
{
    const ScanRegion region = parse_region(argc, argv);
    const uint64_t windows = (uint64_t)region.x_count * (uint64_t)region.z_count;

    printf("v2: scanning %d x %d windows from (%d, %d), seed %llu\n",
           region.x_count, region.z_count, region.x0, region.z0,
           (unsigned long long)region.seed);

    const int32_t threads_per_block = 512;
    const int32_t blocks = (region.z_count + threads_per_block - 1) / threads_per_block;

    int16_t *device_values;
    int32_t *device_x;
    cuda_check(cudaMalloc((void **)&device_values, sizeof(int16_t) * region.z_count), "cudaMalloc values");
    cuda_check(cudaMalloc((void **)&device_x, sizeof(int32_t) * region.z_count), "cudaMalloc x");

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create");
    cuda_check(cudaEventCreate(&stop), "event create");

    cuda_check(cudaEventRecord(start), "event record");
    slime_finder_kernel<<<blocks, threads_per_block>>>(region, device_values, device_x);
    cuda_check(cudaEventRecord(stop), "event record");

    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaEventSynchronize(stop), "kernel execution");

    float kernel_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&kernel_ms, start, stop), "elapsed time");

    int16_t *host_values = (int16_t *)malloc(sizeof(int16_t) * region.z_count);
    int32_t *host_x = (int32_t *)malloc(sizeof(int32_t) * region.z_count);
    cuda_check(cudaMemcpy(host_values, device_values, sizeof(int16_t) * region.z_count, cudaMemcpyDeviceToHost), "copy values");
    cuda_check(cudaMemcpy(host_x, device_x, sizeof(int32_t) * region.z_count, cudaMemcpyDeviceToHost), "copy x");

    BestResult best = {-1, 0, 0};
    for (int32_t i = 0; i < region.z_count; ++i)
    {
        const BestResult cand = {host_values[i], host_x[i], region.z0 + i};
        if (result_better(cand, best))
        {
            best = cand;
        }
    }

    print_best(best);
    print_stats(windows, kernel_ms);

    cudaFree(device_values);
    cudaFree(device_x);
    free(host_values);
    free(host_x);
    return 0;
}
