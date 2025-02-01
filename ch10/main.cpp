#include "kernel.hpp"
#include <iostream>
#include <chrono>

int main()
{
    int len;
    std::cout << "Enter length of array: ";
    std::cin >> len;

    float *in_h = new float[len];
    float *out_h = new float[0];

    for (int i = 0; i < len; i++)
        in_h[i] = 1;

    auto start = std::chrono::high_resolution_clock::now();
    simpleSumReduction(in_h, out_h, len);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diff = end - start;
    std::cout << "Simple Sum Reduction took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    convergentSumReduction(in_h, out_h, len);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Convergent Sum Reduction took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    sharedMemSumReduction(in_h, out_h, len);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Shared Memory Sum Reduction took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    segmenteSumReduction(in_h, out_h, len);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Arbitary Len Sum Reduction took-> " << diff.count() << " ms" << std::endl;

    if (*out_h != len)
    {
        std::cout << "Result Mismatch, Expected: " << len << " | Found: " << *out_h << std::endl;
        goto Here;
    }

    std::cout << "success" << std::endl;
Here:
    delete[] in_h;
    delete[] out_h;

    return 0;
}