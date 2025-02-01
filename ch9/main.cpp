#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "kernel.hpp"

void histogram_sequential(char *input, unsigned int *output, unsigned int n)
{
    for (int i = 0; i < NUM_BINS; i++)
        output[i] = 0;
    for (int i = 0; i < n; i++)
    {
        if (input[i] >= 'a' && input[i] <= 'z')
        {
            int index = (input[i] - 'a') / GAP;
            output[index]++;
        }
    }
}

int main()
{
    std::ifstream file("example.txt");
    if (!file.is_open())
    {
        std::cerr << "Failed to open example text file." << std::endl;
        return 1;
    }

    char ch;
    std::vector<char> lt;
    while (file.get(ch))
    {
        ch = tolower(ch);
        lt.push_back(ch);
    }
    std::cout << "size of text is: " << lt.size() + 1 << std::endl;
    file.close();

    unsigned int n = lt.size();

    char *i_h = new char[n];
    for (int i = 0; i < n; i++)
        i_h[i] = lt[i];
    unsigned int *o_h = new unsigned int[NUM_BINS];

    auto start = std::chrono::high_resolution_clock::now();
    histogram_sequential(i_h, o_h, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diff = end - start;
    std::cout << "Sequential histogram took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    histogram(i_h, o_h, n);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "HIP Basic histogram took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    histogram_private(i_h, o_h, n);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "HIP Private histogram took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    histo_private_shared(i_h, o_h, n);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "HIP Private Shared histogram took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    histo_tc_cont(i_h, o_h, n);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "HIP private Shared with Thread Coarsing (Contiguous) took-> " << diff.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    histo_tc_interl(i_h, o_h, n);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "HIP private Shared with Thread Coarsing (Interleaved) took-> " << diff.count() << " ms" << std::endl;

    for (int i = 0; i < NUM_BINS; i++)
        std::cout << (char)('a' + i * GAP) << "-" << (char)('a' + GAP + i * GAP) << ": " << o_h[i] << std::endl;

    delete[] i_h;
    delete[] o_h;

    return 0;
}
// Correct output for 'example.txt' file
// a-e: 3257450
// e-i: 4708800
// i-m: 2605100
// m-q: 4060475
// q-u: 4571025
// u-y: 1506100
// y-}: 568000
