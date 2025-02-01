#include "kernel.hpp"

__global__ void histogram_kernel(char *input, unsigned int *output, unsigned int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        if (input[index] >= 'a' && input[index] <= 'z')
        {
            int i = (input[index] - 'a') / GAP;
            atomicAdd(&output[i], 1);
        }
    }
}

void histogram(char *input, unsigned int *output, unsigned int n)
{
    unsigned int *d_output;
    char *d_input;
    hipMalloc(&d_output, (26 + GAP - 1) / GAP * sizeof(unsigned int));
    hipMalloc(&d_input, n * sizeof(char));
    for (int i = 0; i < (26 + GAP - 1) / GAP; i++)
        d_output[i] = 0;

    hipMemcpy(d_input, input, n * sizeof(char), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(histogram_kernel, dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0, d_input, d_output, n);

    hipMemcpy(output, d_output, (26 + GAP - 1) / GAP * sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipFree(d_output);
    hipFree(d_input);
}