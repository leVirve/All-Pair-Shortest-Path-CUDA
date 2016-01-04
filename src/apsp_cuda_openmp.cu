#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 32
// TODO: variable block size

int n, m, block_size;
int *r_dist;

__global__
void kernel_phase1(int round, int n, int* dist)
{
    __shared__ int shared_dist[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x,
        y = threadIdx.y,
        i = x + round * BLOCK_SIZE,
        j = y + round * BLOCK_SIZE;

    shared_dist[x][y] = (i < n && j < n) ? dist[i * n + j] : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int tmp = shared_dist[x][k] + shared_dist[k][y];
        if (tmp < shared_dist[x][y]) shared_dist[x][y] = tmp;
        __syncthreads();
    }
    if (i < n && j < n) dist[i * n + j] = shared_dist[x][y];
}

__global__
void kernel_phase2(int round, int n, int* dist)
{
    if (blockIdx.x == round) return;

    __shared__ int shared_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_dist[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x,
        y = threadIdx.y,
        i = x + round * BLOCK_SIZE,
        j = y + round * BLOCK_SIZE;

    shared_pivot[x][y] = (i < n && j < n) ? dist[i * n + j] : INF;

    if (blockIdx.y == 0)
        j = y + blockIdx.x * BLOCK_SIZE;
    else
        i = x + blockIdx.x * BLOCK_SIZE;

    shared_dist[x][y] = (i < n && j < n) ? dist[i * n + j] : INF;
    __syncthreads();

    if (blockIdx.y == 1) {
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int tmp = shared_dist[x][k] + shared_pivot[k][y];
            if (tmp < shared_dist[x][y]) shared_dist[x][y] = tmp;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int tmp = shared_pivot[x][k] + shared_dist[k][y];
            if (tmp < shared_dist[x][y]) shared_dist[x][y] = tmp;
        }
    }

    if (i < n && j < n) dist[i * n + j] = shared_dist[x][y];
}

__global__
void kernel_phase3(int round, int n, int* dist, int offset_lines)
{
    int blockIdx_x = blockIdx.x + offset_lines,
        blockIdx_y = blockIdx.y;
    if (blockIdx_x == round || blockIdx_y == round) return;

    __shared__ int shared_pivot_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_pivot_col[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x,
        y = threadIdx.y,
        i = x + blockIdx_x * BLOCK_SIZE,
        j = y + blockIdx_y * BLOCK_SIZE,
        i_col = y + round * BLOCK_SIZE,
        j_row = x + round * BLOCK_SIZE;

    shared_pivot_row[x][y] = (i < n && i_col < n) ? dist[i * n + i_col] : INF;
    shared_pivot_col[x][y] = (j < n && j_row < n) ? dist[j_row * n + j] : INF;
    __syncthreads();

    if (i >= n || j >= n) return;
    int dij = dist[i * n + j];
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int tmp = shared_pivot_row[x][k] + shared_pivot_col[k][y];
        if (tmp < dij) dij = tmp;
    }
    dist[i * n + j] = dij;
}

void block_FW(int block_size)
{
    float k_time;

    int num_gpus;
    int round = (n + block_size - 1) / block_size;
    ssize_t sz = sizeof(int) * n * n;

    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);

    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3((round + num_gpus - 1) / num_gpus, round);
    dim3 block(block_size, block_size);

    int *device_dist;
    cudaMallocHost(&device_dist, sz);
    cudaMemcpy(device_dist, dist, sz, cudaMemcpyHostToHost);
    r_dist = device_dist;
    #pragma omp parallel
    {
        unsigned int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int r = 0; r < round; ++r) {
            if (thread_id == 0) {
            kernel_phase1<<<grid_phase1, block>>>(r, n, device_dist);

            kernel_phase2<<<grid_phase2, block>>>(r, n, device_dist);
            cudaStreamSynchronize(0);
            }

            #pragma omp barrier

            kernel_phase3<<<grid_phase3, block>>>(r, n, device_dist, thread_id * (round+1) / 2);

            cudaStreamSynchronize(0);
            #pragma omp barrier
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&k_time, start, stop);
        fprintf (stderr, "k_time: %lf\n", k_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main(int argc, char* argv[])
{
    block_size = atoi(argv[3]);

    input(argv[1]);
    block_FW(block_size);
    output(argv[2]);
    cudaFreeHost(r_dist);
    return 0;
}
