#include <stdio.h>
#include <stdlib.h>

const int INF = 10000000;
const int V = 10010;

void block_FW(int B);
__global__ void cal(int* d_dist, int B, int Round, int block_start_x, int block_start_y, int n);

int n, m, B;
int *r_dist;
static int dist[V * V];

void input(char *inFileName)
{
    FILE *infile = fopen(inFileName, "r");
    fscanf(infile, "%d %d", &n, &m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) dist[i *n + j] = (i == j) ? 0 : INF;
    while (--m >= 0) {
        int a, b, v;
        fscanf(infile, "%d %d %d", &a, &b, &v);
        --a, --b;
        dist[a * n + b] = v;
    }
}

void output(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (r_dist[i * n + j] >= INF)  fprintf(outfile, "INF ");
            else fprintf(outfile, "%d ", r_dist[i * n + j]);
        }
        fprintf(outfile, "\n");
    }
}

void init_device()
{
    cudaSetDevice(1);
}

int main(int argc, char* argv[])
{
    B = atoi(argv[3]);
    init_device();

    input(argv[1]);
    block_FW(B);
    output(argv[2]);
    free(r_dist);
    return 0;
}

void launch_cal(
    int* device_dist, int iter,
    int blk_start_x, int blk_start_y,
    int blk_x_sz, int blk_y_sz)
{
    if (!blk_x_sz || !blk_y_sz) return;

    dim3 block(blk_x_sz, blk_y_sz);
    dim3 thread(B, B);
    cal<<<block, thread>>>(device_dist, B, iter, blk_start_x, blk_start_y, n);
}

void block_FW(int B)
{
    float k_time;
    cudaEvent_t start, stop;
    int *device_dist;
    int round = (n + B - 1) / B;
    ssize_t sz = sizeof(int) * n * n;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc(&device_dist, sz);
    cudaMemcpy(device_dist, dist, sz, cudaMemcpyHostToDevice);
    r_dist = (int*) malloc(sz);

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

__global__
void cal(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
    int b_i = blockIdx.x + block_start_x,
        b_j = blockIdx.y + block_start_y,
        i = b_i * B + threadIdx.x,
        j = b_j * B + threadIdx.y;
    if (i >= n) return;
    if (j >= n) return;

    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
        if (dist[i * n + k] + dist[k * n + j] < dist[i * n + j])
            dist[i * n + j] = dist[i * n + k] + dist[k * n + j];
    }
}
