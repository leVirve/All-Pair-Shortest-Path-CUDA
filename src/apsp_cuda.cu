#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int n, m, block_size;
int *r_dist;

__global__
void cal(int* dist, int block_size, int Round, int block_start_x, int block_start_y, int n)
{
    int x = threadIdx.x,
        y = threadIdx.y,
        i = (blockIdx.x + block_start_x) * block_size + x,
        j = (blockIdx.y + block_start_y) * block_size + y;
    if (i >= n) return;
    if (j >= n) return;

    __shared__ int shared_dist[32][32];
    shared_dist[x][y] = (i < n && j < n) ? dist[i * n + j] : INF;

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

    cudaEventRecord(start, 0);
    for (int r = 0; r < round; ++r) {
        // phase1
        launch_cal(device_dist, r, r, r, 1, 1);

        // phase2
        launch_cal(device_dist, r, r, 0, 1, r);
        launch_cal(device_dist, r, r, r + 1, 1, round - r - 1);
        launch_cal(device_dist, r, 0, r, r, 1);
        launch_cal(device_dist, r, r + 1, r, round - r - 1, 1);

        // phase3
        launch_cal(device_dist, r, 0, 0, r, r);
        launch_cal(device_dist, r, 0, r + 1, r, round - r - 1);
        launch_cal(device_dist, r, r + 1, 0, round - r - 1, r);
        launch_cal(device_dist, r, r + 1, r + 1, round - r - 1, round - r - 1);
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
