#include "kernel.hpp"
#include <chrono>
#include <iostream>

void scanOnCPU(double *in, double *out, int n) {
  out[0] = in[0];
  for (int i = 1; i < n; ++i) {
    out[i] = in[i] + out[i - 1];
  }
}

int main() {
  int n;
  std::cout << "Enter the size of array: ";
  std::cin >> n;
  double *arr_in_h = new double[n];
  double *arr_out_h = new double[n];
  double *arr_sout_h = new double[n];

  // Initiating value into test array
  for (int i = 0; i < n; ++i) {
    arr_in_h[i] = 1;
  }

  // Scanning this array using CPU and using the output result to check GPU
  // calculation
  // auto start = std::chrono::high_resolution_clock::now();
  // scanOnCPU(arr_in_h, arr_sout_h, n);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<float, std::milli> diff = end - start;
  // std::cout << "Time taken by the CPU is: " << diff.count() << std::endl;

  // These are implemented only for single Grid, so if you give value of n more
  // than BLOCK_SIZE, It may give wrong answer for error.

  // Kogge_Stone_Scan(arr_in_h, arr_out_h, n);
  // Brent_Kung_Scan(arr_in_h, arr_out_h, n);
  // Coarsed_Kogge_Stone_Scan(arr_in_h, arr_out_h, n);

  // For arbitary Length input.----

  // TODO: Arbitary Length with three kernel for each stage. below is
  // implementation with single kernel using domino effect.

  // Single-pass scan for memory access efficiency
  Single_Pass_Scan(arr_in_h, arr_out_h, n);

  for (int i = 0; i < n; i++) {
    std::cout << i + 1 << "th: " << arr_out_h[i] << " | ";
  }
  std::cout << std::endl;

  delete[] arr_in_h;
  delete[] arr_out_h;
  delete[] arr_sout_h;
}