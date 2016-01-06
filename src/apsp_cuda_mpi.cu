#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "utils.h"

int n, m, block_size;
int *r_dist;
int world_size, rank;

__global__
void kernel_phase1(int round, int n, int* dist, int bsz)
{
    extern __shared__ int shared_dist[];

    int x = threadIdx.x,
        y = threadIdx.y,
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

    int x = threadIdx.x,
        y = threadIdx.y,
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
void kernel_phase3(int round, int n, int* dist, int bsz, int offset_lines)
{
    int blockIdx_x = blockIdx.x + offset_lines,
    blockIdx_y = blockIdx.y;
    if (blockIdx_x == round || blockIdx_y == round) return;

    extern __shared__ int shared_mem[];
    int* shared_pivot_row = &shared_mem[0];
    int* shared_pivot_col = &shared_mem[bsz * bsz];

    int x = threadIdx.x,
        y = threadIdx.y,
        i = x + blockIdx_x * blockDim.x,
        j = y + blockIdx_y * blockDim.y,
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

__global__
void kernel_swap(int* device_dist, int* swap_dist, int offset_lines, int n)
{
    if (blockIdx.x < offset_lines) return;

    int line = blockIdx.x;
    for (int j = 0; j < n; ++j)
        device_dist[line * n + j] = swap_dist[line * n + j];
}

void block_FW(int block_size)
{
    MPI_Status status;
    float k_time;

    int round = (n + block_size - 1) / block_size;
    int offset_blks = (round + world_size - 1) / world_size;
    ssize_t sz = sizeof(int) * n * n;

    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3((round + world_size - 1) / world_size, round);
    dim3 block(block_size, block_size);

    int *device_dist, *swap_dist, *buffer;
    buffer = (int*) malloc(sz);
    cudaSetDevice(rank);
    cudaMalloc(&device_dist, sz);
    if (rank == 0) {
        cudaMalloc(&swap_dist, sz);
        cudaMemcpy(device_dist, dist, sz, cudaMemcpyHostToHost);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int r = 0; r < round; ++r) {

        if (rank == 0) {
            kernel_phase1<<<grid_phase1, block, block_size * block_size * sizeof(int)>>>(r, n, device_dist, block_size);
            kernel_phase2<<<grid_phase2, block, block_size * block_size * sizeof(int) * 2>>>(r, n, device_dist, block_size);
            cudaStreamSynchronize(0);
            if (round > offset_blks) {
                cudaMemcpy(buffer, device_dist, sz, cudaMemcpyDeviceToHost);
                MPI_Send(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            }
        } else if (rank == 1 && round > offset_blks) {
            MPI_Recv(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
            cudaMemcpy(device_dist, buffer, sz, cudaMemcpyHostToDevice);
        }

        if (rank == 0 || (rank == 1 && round > offset_blks))
            kernel_phase3<<<grid_phase3, block, block_size * block_size * sizeof(int) * 2>>>(r, n, device_dist, block_size, offset_blks * rank);
        cudaStreamSynchronize(0);

        if (rank == 0 && round > offset_blks) {
            MPI_Recv(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);
            cudaMemcpy(swap_dist, buffer, sz, cudaMemcpyHostToDevice);
            kernel_swap<<<n, 1>>>(device_dist, swap_dist, block_size * offset_blks, n);
            cudaStreamSynchronize(0);
        } else if (rank == 1 && round > offset_blks) {
            cudaMemcpy(buffer, device_dist, sz, cudaMemcpyDeviceToHost);
            MPI_Send(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&k_time, start, stop);
    fprintf (stderr, "k_time: %lf\n", k_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (rank == 0)
        cudaMemcpy(dist, device_dist, sz, cudaMemcpyDeviceToHost);
    cudaFree(device_dist);
    cudaFree(swap_dist);
    r_dist = dist;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) input(argv[1]);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    block_size = atoi(argv[3]);

    block_FW(block_size);

    if (rank == 0) output(argv[2]);
    cudaFreeHost(r_dist);
    MPI_Finalize();
    return 0;
}
