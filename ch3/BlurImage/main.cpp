#define __HIP_PLATFORM_AMD__
#include "../../lib_/PngHandlib/PNGHand.hpp"
#include <iostream>
#include <string>
#include <hip/hip_runtime.h>

__global__ void blurKernel(unsigned char *i_data, unsigned char *o_data, int h, int w, int blur_size = 3)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    int cnt = 0;
    int pixsm = 0;
    for (int blurRow = -blur_size; blurRow < blur_size + 1; blurRow++)
    {
        for (int blurCol = -blur_size; blurCol < blur_size + 1; blurCol++)
        {
            int ni = i + blurRow;
            int nj = j + 3 * blurCol;
            if (ni >= 0 && ni < h && nj >= 0 && nj < w * 3)
            {
                int idx = nj + ni * w * 3;
                pixsm += i_data[idx];
                cnt++;
            }
        }
    }
    if (cnt == 0)
        cnt = 1;
    o_data[j + i * w * 3] = (unsigned char)(pixsm / cnt);
}

void blurImage(unsigned char *i_data_h, unsigned char *o_data_h, int h, int w, int blur_size = 3)
{
    unsigned char *i_data_d, *o_data_d;
    int n = w * h * 3 * sizeof(unsigned char);

    hipError_t err = hipMalloc(&i_data_d, n);
    if (err != hipSuccess)
    {
        std::cout << "Error: " << hipGetErrorString(err) << std::endl;
    }
    err = hipMalloc(&o_data_d, n);
    if (err != hipSuccess)
    {
        std::cout << "Error: " << hipGetErrorString(err) << std::endl;
    }

    hipMemcpy(i_data_d, i_data_h, n, hipMemcpyHostToDevice);
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((dimBlock.x + 3 * w - 1) / dimBlock.x, (dimBlock.y + h - 1) / dimBlock.y, 1);
    blurKernel<<<dimGrid, dimBlock>>>(i_data_d, o_data_d, h, w, blur_size);
    hipMemcpy(o_data_h, o_data_d, n, hipMemcpyDeviceToHost);

    hipFree(i_data_d);
    hipFree(o_data_d);
}

int main()
{
    std::cout << "Enter the filename: ";
    std::string filename;
    std::cin >> filename;
    PngHand png(filename);
    std::cout << "Enter the blur size: ";
    int blur_size;
    std::cin >> blur_size;
    int width = png.Width();
    int height = png.Height();

    unsigned char *i_data_h = png.Data();
    unsigned char *o_data_h = new unsigned char[width * height * 3];

    blurImage(i_data_h, o_data_h, height, width, blur_size);

    png.Save("blur_" + filename, o_data_h);

    delete[] o_data_h;

    return 0;
}
