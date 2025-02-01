#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#define TILE_WIDTH 32 // See page 116 for to put TILE_WIDTH value at runtime

__global__ void matmulKernel(int *a, int *b, int *c, int x, int y, int z)
{
    int bx = blockIdx.x, tx = threadIdx.x, by = blockIdx.y, ty = threadIdx.y;

    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int pval = 0;
    for (int ph = 0; ph < (y + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
    {
        if (row < x && (ph * TILE_WIDTH + tx) < y)
            Mds[ty][tx] = a[row * y + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0;
        if ((ph * TILE_WIDTH + ty) < y && col < z)
            Nds[ty][tx] = b[(col + ty * z) + (ph * TILE_WIDTH * z)];
        else
            Nds[ty][tx] = 0;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            pval += Mds[ty][i] * Nds[i][tx];
        __syncthreads();
    }

    if (row < x && col < z)
        c[row * z + col] = pval;
}

void matmul(int *a_mat_h, int *b_mat_h, int *c_mat_h, int x, int y, int z)
{
    int n1 = x * y * sizeof(int), n2 = y * z * sizeof(int), n3 = x * z * sizeof(int);
    int *a_mat_d, *b_mat_d, *c_mat_d;

    hipMalloc(&a_mat_d, n1);
    hipMalloc(&b_mat_d, n2);
    hipMalloc(&c_mat_d, n3);

    hipMemcpy(a_mat_d, a_mat_h, n1, hipMemcpyHostToDevice);
    hipMemcpy(b_mat_d, b_mat_h, n2, hipMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((z + dimBlock.x - 1) / dimBlock.x, (x + dimBlock.y - 1) / dimBlock.y, 1);

    matmulKernel<<<dimGrid, dimBlock>>>(a_mat_d, b_mat_d, c_mat_d, x, y, z);

    hipMemcpy(c_mat_h, c_mat_d, n3, hipMemcpyDeviceToHost);

    hipFree(&a_mat_d);
    hipFree(&b_mat_d);
    hipFree(&c_mat_d);
}

void matmul_cpu(int *a_mat_h, int *b_mat_h, int *c_mat_h, int x, int y, int z)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < z; ++j)
        {
            c_mat_h[i * z + j] = 0;
            for (int k = 0; k < y; ++k)
            {
                c_mat_h[i * z + j] += a_mat_h[i * y + k] * b_mat_h[k * z + j];
            }
        }
    }
}

int main()
{
    int x, y, z;
    std::cout << "Enter row dimension of mat_1: ";
    std::cin >> x;
    std::cout << "Enter common dimension of both matrixes: ";
    std::cin >> y;
    std::cout << "Enter column dimension of mat_2: ";
    std::cin >> z;
    int *a_mat_h = new int[x * y];
    int *b_mat_h = new int[y * z];
    int *c_mat_h = new int[x * z];
    int *c_mat_cpu = new int[x * z];
    for (int i = 0; i < x; ++i)
        for (int j = 0; j < y; ++j)
            a_mat_h[i * y + j] = 1;

    for (int i = 0; i < y; ++i)
        for (int j = 0; j < z; ++j)
            b_mat_h[i * z + j] = 1;

    auto start = std::chrono::high_resolution_clock::now();
    matmul(a_mat_h, b_mat_h, c_mat_h, x, y, z);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "GPU Time: " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matmul_cpu(a_mat_h, b_mat_h, c_mat_cpu, x, y, z);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CPU Time: " << duration.count() << " ms" << std::endl;

    for (int i = 0; i < x * z; i++)
    {
        if (c_mat_h[i] != c_mat_cpu[i])
        {
            std::cout << "Mismatch at " << i << " : " << c_mat_h[i] << " " << c_mat_cpu[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success" << std::endl;

    delete[] a_mat_h;
    delete[] b_mat_h;
    delete[] c_mat_h;
    delete[] c_mat_cpu;

    return 0;
}