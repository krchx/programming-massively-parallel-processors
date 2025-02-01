#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
// #define N 2560
#define TILE_WIDTH 2 // See page 116 for various TILE_WIDTH values
int N;

__global__ void matmulKernel(int *a, int *b, int *c, int N)
{
    int bx = blockIdx.x, tx = threadIdx.x, by = blockIdx.y, ty = threadIdx.y;

    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int pval = 0;
    for (int ph = 0; ph < N / TILE_WIDTH; ++ph)
    {
        Mds[ty][tx] = a[row * N + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = b[(col + ty * N) + (ph * TILE_WIDTH * N)];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            pval += Mds[ty][i] * Nds[i][tx];
        __syncthreads();
    }
    c[row * N + col] = pval;
}

void matmul(int *a_mat_h, int *b_mat_h, int *c_mat_h)
{
    int n = N * N * sizeof(int);
    int *a_mat_d, *b_mat_d, *c_mat_d;

    hipMalloc(&a_mat_d, n);
    hipMalloc(&b_mat_d, n);
    hipMalloc(&c_mat_d, n);

    hipMemcpy(a_mat_d, a_mat_h, n, hipMemcpyHostToDevice);
    hipMemcpy(b_mat_d, b_mat_h, n, hipMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y, 1);

    matmulKernel<<<dimGrid, dimBlock>>>(a_mat_d, b_mat_d, c_mat_d, N);

    hipMemcpy(c_mat_h, c_mat_d, n, hipMemcpyDeviceToHost);

    hipFree(&a_mat_d);
    hipFree(&b_mat_d);
    hipFree(&c_mat_d);
}

void matmul_cpu(int *a_mat_h, int *b_mat_h, int *c_mat_h)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; ++j)
        {
            c_mat_h[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
            {
                c_mat_h[i * N + j] += a_mat_h[i * N + k] * b_mat_h[k * N + j];
            }
        }
    }
}

int solve()
{
    int *a_mat_h = new int[N * N];
    int *b_mat_h = new int[N * N];
    int *c_mat_h = new int[N * N];
    int *c_mat_cpu = new int[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; ++j)
        {
            a_mat_h[i * N + j] = i + j;
            b_mat_h[i * N + j] = i - j;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    matmul(a_mat_h, b_mat_h, c_mat_h);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "GPU Time: " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matmul_cpu(a_mat_h, b_mat_h, c_mat_cpu);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CPU Time: " << duration.count() << " ms" << std::endl;

    for (int i = 0; i < N * N; i++)
    {
        if (c_mat_h[i] != c_mat_cpu[i])
        {
            std::cout << "Mismatch at " << i << " : " << c_mat_h[i] << " " << c_mat_cpu[i] << std::endl;
            return 0;
        }
    }
    std::cout << "Success" << std::endl;

    delete[] a_mat_h;
    delete[] b_mat_h;
    delete[] c_mat_h;
    delete[] c_mat_cpu;

    return 1;
}
int main()
{
    for (int i = 2; i < 1100; i = i * 2)
    {
        N = i;
        int a = solve();
        if (a == 0)
            break;
    }
}