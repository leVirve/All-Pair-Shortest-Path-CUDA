#define HANDLE_ERROR(status) \
    if (status != cudaSuccess) { \
        fprintf(stderr, "Line%d: %s\n", __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    }
