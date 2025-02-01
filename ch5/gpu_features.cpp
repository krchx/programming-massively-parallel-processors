#define __HIP_PLATFORM_AMD__
#include <iostream>
#include <hip/hip_runtime.h>

int main()
{
    hipDeviceProp_t prop;
    int count;
    hipGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "Shared Memory per SM:" << prop.sharedMemPerBlock << std::endl;

        std::cout << "********************************************************************" << std::endl;
    }
    return 0;
}