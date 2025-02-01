#include <iostream>
#include <hip/hip_runtime.h>
#include <chrono>
#include "../../lib_/PngHandlib/PNGHand.hpp"

__global__ void greyScaleConvKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < width && j < height)
    {
        int idx = j * width + i;
        output[idx] = 0.299f * input[3 * idx] + 0.587f * input[3 * idx + 1] + 0.114f * input[3 * idx + 2];
    }
}

void greyScaleConv(unsigned char *input_h, unsigned char *output_h, int width, int height)
{
    unsigned char *input_d, *output_d;
    hipMalloc(&input_d, 3 * width * height * sizeof(unsigned char));
    hipMalloc(&output_d, width * height * sizeof(unsigned char));

    hipMemcpy(input_d, input_h, 3 * width * height * sizeof(unsigned char), hipMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);

    greyScaleConvKernel<<<dimGrid, dimBlock>>>(input_d, output_d, width, height);

    hipMemcpy(output_h, output_d, width * height * sizeof(unsigned char), hipMemcpyDeviceToHost);

    hipFree(input_d);
    hipFree(output_d);
}

void greyScaleConvCPU(unsigned char *input_h, unsigned char *output_h, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int idx = j * width + i;
            output_h[idx] = 0.299f * input_h[3 * idx] + 0.587f * input_h[3 * idx + 1] + 0.114f * input_h[3 * idx + 2];
        }
    }
}

int main()
{

    PngHand png("image.png");
    int width = png.Width();
    int height = png.Height();

    unsigned char *input_h = png.Data();
    unsigned char *output_h = new unsigned char[width * height];
    unsigned char *output_h_cpu = new unsigned char[width * height];

    auto start = std::chrono::high_resolution_clock::now();
    greyScaleConv(input_h, output_h, width, height);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Parallel Time: " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    greyScaleConvCPU(input_h, output_h_cpu, width, height);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU Time: " << elapsed.count() << " ms" << std::endl;

    for (int i = 0; i < width * height; i++)
    {
        if (output_h[i] != output_h_cpu[i])
        {
            std::cout << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    png.SaveGrey("grey.png", output_h);

    delete[] output_h;
    delete[] output_h_cpu;

    return 0;
}