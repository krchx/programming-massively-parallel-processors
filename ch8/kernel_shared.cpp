#include "stencil_kernel.hpp"
#define OUT_TILE_DIM 6
#define IN_TILE_DIM 8

__constant__ double C[6 * RADIUS + 1];

__global__ void stencil_shared_kernel(double *A, double *B, int n)
{
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - RADIUS;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;

    __shared__ float A_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n)
        A_s[threadIdx.z][threadIdx.y][threadIdx.x] = A[i * n * n + j * n + k];
    __syncthreads();

    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1)
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM)
            B[i * n * n + j * n + k] = C[0] * A_s[threadIdx.z][threadIdx.y][threadIdx.x] + C[1] * A_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] + C[2] * A_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] + C[3] * A_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] + C[4] * A_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] + C[5] * A_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] + C[6] * A_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
}

void stencil_shared(double *i_h, double *o_h, double *c_h, int n)
{
    int sz = n * n * n * sizeof(double);

    double *i_d, *o_d;
    hipMalloc(&i_d, sz);
    hipMalloc(&o_d, sz);

    hipMemcpy(i_d, i_h, sz, hipMemcpyHostToDevice);
    hipMemcpy(o_d, i_h, sz, hipMemcpyHostToDevice); // To get corner values on output
    hipMemcpyToSymbol(C, c_h, (6 * RADIUS + 1) * sizeof(double));

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((n + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (n + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (n + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_shared_kernel<<<dimGrid, dimBlock>>>(i_d, o_d, n);

    hipMemcpy(o_h, o_d, sz, hipMemcpyDeviceToHost);

    hipFree(i_d);
    hipFree(o_d);
}