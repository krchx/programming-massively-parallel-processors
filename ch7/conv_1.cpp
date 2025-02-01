#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <hip/hip_runtime.h>
#include <chrono>
#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void conv_2D_tiled_kernel(float *A, float *B, int ht, int wd)
{
    int col = (threadIdx.x - FILTER_RADIUS) + blockIdx.x * OUT_TILE_DIM;
    int row = (threadIdx.y - FILTER_RADIUS) + blockIdx.y * OUT_TILE_DIM;

    // if (row >= 0 && row < 2 * FILTER_RADIUS + 1 && col >= 0 && col < 2 * FILTER_RADIUS + 1)
    //     B[col + row * wd] = F[row][col];

    __shared__ float A_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < ht && col >= 0 && col < wd)
        A_s[threadIdx.y][threadIdx.x] = A[row * wd + col];
    else
        A_s[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();

    int tcol = threadIdx.x - FILTER_RADIUS;
    int trow = threadIdx.y - FILTER_RADIUS;
    if (row >= 0 && row < ht && col >= 0 && col < wd)
    {
        if (trow >= 0 && trow < OUT_TILE_DIM && tcol >= 0 && tcol < OUT_TILE_DIM)
        {
            float val = 0.0;
            for (int i = 0; i < FILTER_RADIUS * 2 + 1; ++i)
                for (int j = 0; j < FILTER_RADIUS * 2 + 1; ++j)
                    val += F[i][j] * A_s[i + trow][j + tcol];
            B[col + row * wd] = val;
        }
    }
}

void convolution_2D(float *i_h, float *o_h, float *f_h, int ht, int wd, int fdim)
{
    float *i_d, *o_d;
    hipMalloc(&i_d, ht * wd * sizeof(float));
    hipMalloc(&o_d, ht * wd * sizeof(float));

    hipMemcpy(i_d, i_h, ht * wd * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpyToSymbol(F, f_h, fdim * fdim * sizeof(float));

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 dimgrid((wd + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (ht + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 1);
    conv_2D_tiled_kernel<<<dimgrid, dimBlock>>>(i_d, o_d, ht, wd);

    hipMemcpy(o_h, o_d, ht * wd * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(i_d);
    hipFree(o_d);
}

void convolution_2D_cpu(float *i_h, float *o_h, float *f_h, int ht, int wd, int fdim)
{
    for (int i = 0; i < ht; ++i)
    {
        for (int j = 0; j < wd; ++j)
        {
            float val = 0.0f;
            for (int k = 0; k < fdim; ++k)
            {
                for (int l = 0; l < fdim; ++l)
                {
                    int x = j + l - FILTER_RADIUS;
                    int y = i + k - FILTER_RADIUS;
                    if (x >= 0 && x < wd && y >= 0 && y < ht)
                        val += f_h[k * fdim + l] * i_h[x + y * wd];
                }
            }
            o_h[i * wd + j] = val;
        }
    }
}

int main()
{
    int ht, wd;
    int fdim = 2 * FILTER_RADIUS + 1;
    std::cout << "Enter dim of sq. matrix: ";
    std::cin >> ht >> wd;

    float *i_h = new float[ht * wd];
    float *o_h = new float[ht * wd];
    float *o_h_cpu = new float[ht * wd];
    float *f_h = new float[fdim * fdim];

    for (int i = 0; i < ht * wd; ++i)
        i_h[i] = i;
    for (int i = 0; i < fdim * fdim; ++i)
        f_h[i] = 0.01 * i;

    auto start = std::chrono::high_resolution_clock::now();
    convolution_2D(i_h, o_h, f_h, ht, wd, fdim);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "GPU TIME: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    convolution_2D_cpu(i_h, o_h_cpu, f_h, ht, wd, fdim);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CPU TIME: " << duration.count() << "ms" << std::endl;

    for (int i = 0; i < ht * wd; ++i)
    {
        if (o_h[i] != o_h_cpu[i])
        {
            std::cout << i << ": " << o_h[i] << ", " << o_h_cpu[i] << std::endl;
            goto Here;
        }
    }

    std::cout << "Success" << std::endl;

Here:
    delete[] i_h;
    delete[] o_h;
    delete[] o_h_cpu;
    delete[] f_h;

    return 0;
}