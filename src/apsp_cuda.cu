#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define BLOCK_SIZE 32

int n, m, block_size;
int *r_dist;

__global__
void cal(int* dist, int block_size, int Round, int block_start_x, int block_start_y, int n)
{
    int i = (blockIdx.x + block_start_x) * block_size + threadIdx.x,
        j = (blockIdx.y + block_start_y) * block_size + threadIdx.y;
    if (i >= n) return;
    if (j >= n) return;

    for (int k = Round * block_size; k < (Round + 1) * block_size && k < n; ++k) {
        int &dik = dist[i * n + k],
            &dkj = dist[k * n + j],
            &dij = dist[i * n + j];
        if (dik + dkj < dij) dij = dik + dkj;
    }
}

void launch_cal(
    int* device_dist, int iter,
    int blk_start_x, int blk_start_y,
    int blk_x_sz, int blk_y_sz)
{
    if (!blk_x_sz || !blk_y_sz) return;

    dim3 block(blk_x_sz, blk_y_sz);
    dim3 thread(block_size, block_size);
    cal<<<block, thread>>>(device_dist, block_size, iter, blk_start_x, blk_start_y, n);
}

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
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int tmp = shared_dist[x][k] + shared_pivot[k][y];
            if (tmp < shared_dist[x][y]) shared_dist[x][y] = tmp;
        }
    } else {
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int tmp = shared_pivot[x][k] + shared_dist[k][y];
            if (tmp < shared_dist[x][y]) shared_dist[x][y] = tmp;
        }
    }

    if (i < n && j < n) dist[i * n + j] = shared_dist[x][y];
}

__global__
void kernel_phase3(int round, int n, int* dist)
{
    if (blockIdx.x == round || blockIdx.y == round) return;

    __shared__ int shared_pivot_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_pivot_col[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x,
        y = threadIdx.y,
        i = x + blockIdx.x * blockDim.x,
        j = y + blockIdx.y * blockDim.y,
        i_col = y + round * BLOCK_SIZE,
        j_row = x + round * BLOCK_SIZE;

#ifdef PLAIN_MEM
    if (i >= n) return;
    if (j >= n) return;

    for (int k = round * BLOCK_SIZE; k < (round + 1) * BLOCK_SIZE && k < n; ++k) {
        int &dik = dist[i * n + k],
            &dkj = dist[k * n + j],
            &dij = dist[i * n + j];
        if (dik + dkj < dij) dij = dik + dkj;
    }
#else
    shared_pivot_row[x][y] = (i < n && i_col < n) ? dist[i * n + i_col] : INF;
    shared_pivot_col[x][y] = (j < n && j_row < n) ? dist[j_row * n + j] : INF;
    __syncthreads();

    if (i >= n || j >= n) return;
    int dij = dist[i * n + j];
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int tmp = shared_pivot_row[x][k] + shared_pivot_col[k][y];
       if (tmp < dij) dij = tmp;
    }
    dist[i * n + j] = dij;
#endif
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
    cudaMemcpy(device_dist, dist, sz, cudaMemcpyHostToDevice);
    r_dist = (int*) malloc(sz);

    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3(round, round);
    dim3 block(block_size, block_size);

    cudaEventRecord(start, 0);
    for (int r = 0; r < round; ++r) {
        kernel_phase1<<<grid_phase1, block>>>(r, n, device_dist);

        kernel_phase2<<<grid_phase2, block>>>(r, n, device_dist);

#ifdef PLAIN_MEM
        launch_cal(device_dist, r, 0, 0, r, r);
        launch_cal(device_dist, r, 0, r + 1, r, round - r - 1);
        launch_cal(device_dist, r, r + 1, 0, round - r - 1, r);
        launch_cal(device_dist, r, r + 1, r + 1, round - r - 1, round - r - 1);
#else
        kernel_phase3<<<grid_phase3, block>>>(r, n, device_dist);
#endif
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&k_time, start, stop);
    cudaMemcpy(r_dist, device_dist, sz, cudaMemcpyDeviceToHost);
    cudaFree(device_dist);

    fprintf (stderr, "k_time: %lf\n", k_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{
    block_size = atoi(argv[3]);

    input(argv[1]);
    block_FW(block_size);
    output(argv[2]);
    free(r_dist);
    return 0;
}
