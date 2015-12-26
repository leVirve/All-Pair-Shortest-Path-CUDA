#define HANDLE_ERROR(status) \
    if (status != cudaSuccess) { \
        fprintf(stderr, "Line%d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    }


const int INF = 10000000;
const int V = 10010;
extern int n, m, *r_dist;;
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
