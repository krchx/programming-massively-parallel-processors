#include "stencil_kernel.hpp"
#define IN_TILE_DIM 32
#define OUT_TILE_DIM 30

__constant__ double C[6 * RADIUS + 1];

__global__ void stencil_tc_kernel(double *A, double *B, int n)
{
    int iSt = blockIdx.z * OUT_TILE_DIM;
    int j = threadIdx.y - RADIUS + blockIdx.y * OUT_TILE_DIM;
    int k = threadIdx.x - RADIUS + blockIdx.x * OUT_TILE_DIM;

    __shared__ double prev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ double curr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ double next_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iSt >= 1 && iSt <= n && j >= 0 && j < n && k >= 0 && k < n)
        prev_s[threadIdx.y][threadIdx.x] = A[(iSt - 1) * n * n + j * n + k];

    if (iSt >= 0 && iSt < n && j >= 0 && j < n && k >= 0 && k < n)
        curr_s[threadIdx.y][threadIdx.x] = A[iSt * n * n + j * n * k];

    for (int i = iSt; i < iSt + OUT_TILE_DIM; ++i)
    {
        if (i >= 0 && i < n - 1 && j >= 0 && j < n && k >= 0 && k < n)
            next_s[threadIdx.y][threadIdx.x] = A[(i + 1) * n * n + j * n + k];
        __syncthreads();

        if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1)
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1)
                B[i * n * n + j * n + k] = curr_s[5][5];
        // C[0] * curr_s[threadIdx.y][threadIdx.x];
        // C[1] * curr_s[threadIdx.y][threadIdx.x - 1] +
        // C[2] * curr_s[threadIdx.y][threadIdx.x + 1] +
        // C[3] * curr_s[threadIdx.y + 1][threadIdx.x] +
        // C[4] * curr_s[threadIdx.y - 1][threadIdx.x] +
        // C[5] * prev_s[threadIdx.y][threadIdx.x] +
        // C[6] * next_s[threadIdx.y][threadIdx.x];
        __syncthreads();

        prev_s[threadIdx.y][threadIdx.x] = curr_s[threadIdx.y][threadIdx.x];
        curr_s[threadIdx.y][threadIdx.x] = next_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_tc(double *i_h, double *o_h, double *c_h, int n)
{
    int sz = n * n * n * sizeof(double);

    double *i_d, *o_d;
    hipMalloc(&i_d, sz);
    hipMalloc(&o_d, sz);

    hipMemcpy(i_d, i_h, sz, hipMemcpyHostToDevice);
    hipMemcpy(o_d, i_h, sz, hipMemcpyHostToDevice); // To get corner values on output
    hipMemcpyToSymbol(C, c_h, (6 * RADIUS + 1) * sizeof(double));

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 dimGrid((n + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (n + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (n + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_tc_kernel<<<dimGrid, dimBlock>>>(i_d, o_d, n);

    hipMemcpy(o_h, o_d, sz, hipMemcpyDeviceToHost);

    hipFree(i_d);
    hipFree(o_d);
}