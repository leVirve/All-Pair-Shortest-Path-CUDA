#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // for omp_get_wtime
#include "utils.h"

int n, m, block_size;
int *r_dist;

__global__
void kernel_phase1(int round, int n, int* dist, int bsz)
{
    extern __shared__ int shared_dist[];

    int y = threadIdx.x,
        x = threadIdx.y,
        i = x + round * bsz,
        j = y + round * bsz;

    shared_dist[x * bsz + y] = (i < n && j < n) ? dist[i * n + j] : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < bsz; ++k) {
        int tmp = shared_dist[x * bsz + k] + shared_dist[k * bsz + y];
        if (tmp < shared_dist[x * bsz + y]) shared_dist[x * bsz + y] = tmp;
        __syncthreads();
    }
    if (i < n && j < n) dist[i * n + j] = shared_dist[x * bsz + y];
    __syncthreads();
}

__global__
void kernel_phase2(int round, int n, int* dist, int bsz)
{
    if (blockIdx.x == round) return;

    extern __shared__ int shared_mem[];
    int* shared_pivot = &shared_mem[0];
    int* shared_dist = &shared_mem[bsz * bsz];

    int y = threadIdx.x,
        x = threadIdx.y,
        i = x + round * bsz,
        j = y + round * bsz;

    shared_pivot[x * bsz + y] = (i < n && j < n) ? dist[i * n + j] : INF;

    if (blockIdx.y == 0)
        j = y + blockIdx.x * bsz;
    else
        i = x + blockIdx.x * bsz;

    if (i >= n || j >= n) return;
    shared_dist[x * bsz + y] = (i < n && j < n) ? dist[i * n + j] : INF;
    __syncthreads();

    if (blockIdx.y == 1) {
        #pragma unroll
        for (int k = 0; k < bsz; ++k) {
            int tmp = shared_dist[x * bsz + k] + shared_pivot[k * bsz + y];
            if (tmp < shared_dist[x * bsz + y]) shared_dist[x * bsz + y] = tmp;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < bsz; ++k) {
            int tmp = shared_pivot[x * bsz + k] + shared_dist[k * bsz + y];
            if (tmp < shared_dist[x * bsz + y]) shared_dist[x * bsz + y] = tmp;
        }
    }

    if (i < n && j < n) dist[i * n + j] = shared_dist[x * bsz + y];
}

__global__
void kernel_phase3(int round, int n, int* dist, int bsz)
{
    if (blockIdx.x == round || blockIdx.y == round) return;

    extern __shared__ int shared_mem[];
    int* shared_pivot_row = &shared_mem[0];
    int* shared_pivot_col = &shared_mem[bsz * bsz];

    int y = threadIdx.x,
        x = threadIdx.y,
        i = x + blockIdx.x * blockDim.x,
        j = y + blockIdx.y * blockDim.y,
        i_col = y + round * bsz,
        j_row = x + round * bsz;

    shared_pivot_row[x * bsz + y] = (i < n && i_col < n) ? dist[i * n + i_col] : INF;
    shared_pivot_col[x * bsz + y] = (j < n && j_row < n) ? dist[j_row * n + j] : INF;
    __syncthreads();

    if (i >= n || j >= n) return;
    int dij = dist[i * n + j];
    #pragma unroll
    for (int k = 0; k < bsz; ++k) {
        int tmp = shared_pivot_row[x * bsz + k] + shared_pivot_col[k * bsz + y];
        if (tmp < dij) dij = tmp;
    }
    dist[i * n + j] = dij;
}

void block_FW(int block_size)
{
    float k_time;

    int *device_dist;
    int round = (n + block_size - 1) / block_size;
    ssize_t sz = sizeof(int) * n * n;
    cudaEvent_t start, stop;

    cudaSetDevice(1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&device_dist, sz);
    cudaMemcpyAsync(device_dist, dist, sz, cudaMemcpyHostToDevice);

    r_dist = (int*) malloc(sz);

    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3(round, round);
    dim3 block(block_size, block_size);

    cudaEventRecord(start, 0);
    for (int r = 0; r < round; ++r) {
        kernel_phase1<<<grid_phase1, block, block_size * block_size * sizeof(int)>>>(r, n, device_dist, block_size);

        kernel_phase2<<<grid_phase2, block, block_size * block_size * sizeof(int) * 2>>>(r, n, device_dist, block_size);

        kernel_phase3<<<grid_phase3, block, block_size * block_size * sizeof(int) * 2>>>(r, n, device_dist, block_size);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&k_time, start, stop);

    cudaMemcpy(r_dist, device_dist, sz, cudaMemcpyDeviceToHost);
    fprintf (stderr, "k_time: %lf ms\n", k_time);\
    cudaFree(device_dist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{
    block_size = atoi(argv[3]);

    double io_time = 0, s = omp_get_wtime();
    input(argv[1]);
    io_time += omp_get_wtime() - s;

    block_FW(block_size);

    s = omp_get_wtime();
    output(argv[2]);
    io_time += omp_get_wtime() - s;
    printf("io_time: %lf sec\n", io_time);

    free(r_dist);
    return 0;
}
