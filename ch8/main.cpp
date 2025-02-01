// A 7-point 3D stencil
#include <iostream>
#include "stencil_kernel.hpp"

int main()
{
    int n;
    std::cout << "Enter Tensor Dim: ";
    std::cin >> n;

    double *i_t = new double[n * n * n];
    double *o_t1 = new double[n * n * n];
    double *o_t2 = new double[n * n * n];
    double *c_h = new double[6 * RADIUS + 1];

    for (int i = 0; i < n * n * n; ++i)
        i_t[i] = 1;
    for (int i = 0; i < 6 * RADIUS + 1; ++i)
        c_h[i] = 1;
    // c_h[3] = 1;

    // stencil_shared(i_t, o_t1, c_h, n);
    stencil_tc(i_t, o_t2, c_h, n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
                if (o_t1[i * n * n + j * n + k] != o_t2[i * n * n + j * n + k])
                {
                    std::cout << "Found unequal @ " << i << " " << j << " " << k << " -> " << o_t1[i * n * n + j * n + k] << " ~ " << o_t2[i * n * n + j * n + k] << std::endl;
                    goto Here;
                }
            // std::cout << o_t[i * n * n + j * n + k] << " ";
            // std::cout << std::endl;
        }
        // std::cout << "*****************" << std::endl;
    }

Here:
    delete[] i_t;
    delete[] o_t1;
    delete[] o_t2;
    delete[] c_h;

    return 0;
}