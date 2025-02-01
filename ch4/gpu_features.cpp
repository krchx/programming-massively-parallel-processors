#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int main()
{
    hipDeviceProp_t prop;
    int count;
    hipGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "max Threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate << std::endl;
        std::cout << "Max Threads per block along X: " << prop.maxThreadsDim[0] << " Y: " << prop.maxThreadsDim[1] << " Z: " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max Grid size along X: " << prop.maxGridSize[0] << " Y: " << prop.maxGridSize[1] << " Z: " << prop.maxGridSize[2] << std::endl;
        std::cout << "Max Registers available per SM: " << prop.regsPerBlock << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "max Threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;

        std::cout << "********************************************************************" << std::endl;
    }
}