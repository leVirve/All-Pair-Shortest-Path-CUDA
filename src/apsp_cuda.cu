#include <stdio.h>
#include <stdlib.h>

const int INF = 10000000;
const int V = 10010;

void block_FW(int B);
__global__ void cal(int* d_dist, int B, int Round, int block_start_x, int block_start_y, int n);

int n, m;   // Number of vertices, edges
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
    int B = atoi(argv[3]);
    init_device();

    input(argv[1]);
    block_FW(B);
    output(argv[2]);
    free(r_dist);
    return 0;
}

void block_FW(int B)
{
    int *d_dist;
    ssize_t sz = sizeof(int) * n * n;

    cudaMalloc(&d_dist, sz);
    cudaMemcpy(d_dist, dist, sz, cudaMemcpyHostToDevice);
    r_dist = (int*) malloc(sz);

    cudaMemcpy(r_dist, d_dist, sz, cudaMemcpyDeviceToHost);
    int round = (n + B - 1) / B;
    for (int r = 0; r < round; ++r) {
        // phase1
        cal<<<1, 1>>>(d_dist, B, r, r, r, n);

        // phase2
        cal<<<1, r>>>(d_dist, B, r, r, 0, n);
        cal<<<1, round - r - 1>>>(d_dist, B, r, r, r + 1, n);
        cal<<<r, 1>>>(d_dist, B, r, 0, r, n);
        cal<<<round - r - 1, 1>>>(d_dist, B, r, r + 1, r, n);

        // phase3
        cal<<<r, r>>>(d_dist, B, r, 0, 0, n);
        cal<<<r, round - r - 1>>>(d_dist, B, r, 0, r + 1, n);
        cal<<<round - r - 1, r>>>(d_dist, B, r, r + 1, 0, n);
        cal<<<round - r - 1, round - r - 1>>>(d_dist, B, r, r + 1, r + 1, n);
    }
    cudaMemcpy(r_dist, d_dist, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
}

__global__
void cal(int* dist, int B, int Round, int block_start_x, int block_start_y, int n)
{
    // blockIdx.y
    int b_i = blockIdx.x + block_start_x, b_j = threadIdx.x + block_start_y;
    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
        int block_internal_start_x = b_i * B;
        int block_internal_start_y = b_j * B;
        int block_internal_end_x   = (b_i + 1) * B;
        int block_internal_end_y   = (b_j + 1) * B;

        if (block_internal_end_x > n)   block_internal_end_x = n;
        if (block_internal_end_y > n)   block_internal_end_y = n;

        for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
            for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                if (dist[i * n + k] + dist[k * n + j] < dist[i * n + j])
                    dist[i * n + j] = dist[i * n + k] + dist[k * n + j];
            }
        }
    }
}
