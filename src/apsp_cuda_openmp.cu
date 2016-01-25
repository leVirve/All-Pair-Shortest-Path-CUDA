#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
void kernel_phase3(int round, int n, int* dist, int bsz, int offset_lines)
{
    int blockIdx_x = blockIdx.x + offset_lines,
    blockIdx_y = blockIdx.y;
    if (blockIdx_x == round || blockIdx_y == round) return;

    extern __shared__ int shared_mem[];
    int* shared_pivot_row = &shared_mem[0];
    int* shared_pivot_col = &shared_mem[bsz * bsz];

    int y = threadIdx.x,
        x = threadIdx.y,
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
    int blockIdx_x = blockIdx.x + offset_lines;
    int y = threadIdx.x,
        x = threadIdx.y,
        i = x + blockIdx_x * blockDim.x,
        j = y + blockIdx.y * blockDim.y;
    if (i >= n || j >= n) return;
    device_dist[i * n + j] = swap_dist[i * n + j];
}

void block_FW(int block_size)
{
    int num_gpus;
    float gtmp, m_time = 0;
    int round = (n + block_size - 1) / block_size;
    size_t sz = sizeof(int) * n * n;
    cudaEvent_t start, stop;

    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);
    int offset_blks = (round + num_gpus - 1) / num_gpus;

    dim3 grid_phase1(1, 1);
    dim3 grid_phase2(round, 2);
    dim3 grid_phase3((round + num_gpus - 1) / num_gpus, round);
    dim3 block(block_size, block_size);

    int *dev_dist[num_gpus], *swap_dist;

    #pragma omp parallel
    {
        float k_time, tmp;
        unsigned int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        HANDLE_ERROR(cudaMalloc(&dev_dist[thread_id], sz));
        if (thread_id == 0) {
            HANDLE_ERROR(cudaMalloc(&swap_dist, sz));
            cudaEventRecord(start, 0);
            cudaMemcpy(dev_dist[0], dist, sz, cudaMemcpyHostToDevice);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&tmp, start, stop);
            m_time += tmp;
        }

        cudaEventRecord(start, 0);
        for (int r = 0; r < round; ++r) {

            #pragma omp barrier /* don't know why this barrier can esape segmentation fault... */

            if (thread_id == 0) {
                kernel_phase1<<<grid_phase1, block, block_size * block_size * sizeof(int)>>>(r, n, dev_dist[thread_id], block_size);
                kernel_phase2<<<grid_phase2, block, block_size * block_size * sizeof(int) * 2>>>(r, n, dev_dist[thread_id], block_size);
                if (round > offset_blks) {
                    cudaEventRecord(start, 0);
                    HANDLE_ERROR(cudaMemcpy(dev_dist[1], dev_dist[0], sz, cudaMemcpyDefault));
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&tmp, start, stop);
                    m_time += tmp;
                }
                // puts("danger* segmentation fault *_*");
                cudaStreamSynchronize(0);
            }

            #pragma omp barrier

            if (thread_id == 0 || (thread_id == 1 && round > offset_blks))
                kernel_phase3<<<grid_phase3, block, block_size * block_size * sizeof(int) * 2>>>(r, n, dev_dist[thread_id], block_size, offset_blks * thread_id);
            cudaStreamSynchronize(0);

            if (thread_id == 0 && round > offset_blks) {
                cudaEventRecord(start, 0);
                cudaMemcpy(swap_dist, dev_dist[1], sz, cudaMemcpyDefault);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&tmp, start, stop);
                m_time += tmp;
                kernel_swap<<<grid_phase3, block>>>(dev_dist[0], swap_dist, offset_blks, n);
                cudaStreamSynchronize(0);
            }
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&k_time, start, stop);
        fprintf (stderr, "k_time: %lf ms\n", k_time);
        fprintf (stderr, "m_time: %lf ms\n", m_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaEventRecord(start, 0);
    cudaMemcpy(dist, dev_dist[0], sz, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gtmp, start, stop);
    m_time += gtmp;


    cudaFree(dev_dist[0]);
    cudaFree(dev_dist[1]);
    cudaFree(swap_dist);
    r_dist = dist;
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

    return 0;
}
