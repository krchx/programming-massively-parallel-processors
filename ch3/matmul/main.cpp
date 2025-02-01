#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <iostream>
#define BLOCK_SIZE 16

__global__ void matmulKernel(int *a, int *b, int *c, int i, int j, int k) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < i && col < k) {
    int val = 0;
    for (int m = 0; m < j; ++m)
      val += a[m + row * j] * b[col + m * k];
    c[col + row * k] = val;
  }
}

void matmul(int *a_mat_h, int *b_mat_h, int *c_mat_h, int i, int j, int k) {
  int *a_mat_d, *b_mat_d, *c_mat_d;
  int as = i * j * sizeof(int);
  int bs = j * k * sizeof(int);
  int cs = i * k * sizeof(int);

  hipMalloc(&a_mat_d, as);
  hipMalloc(&b_mat_d, bs);
  hipMalloc(&c_mat_d, cs);

  hipMemcpy(a_mat_d, a_mat_h, as, hipMemcpyHostToDevice);
  hipMemcpy(b_mat_d, b_mat_h, bs, hipMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x,
               (i + dimBlock.y - 1) / dimBlock.y, 1);

  matmulKernel<<<dimGrid, dimBlock>>>(a_mat_d, b_mat_d, c_mat_d, i, j, k);

  hipMemcpy(c_mat_h, c_mat_d, cs, hipMemcpyDeviceToHost);

  hipFree(a_mat_d);
  hipFree(b_mat_d);
  hipFree(c_mat_d);

  return;
}

void matmul_cpu(int *a_mat_h, int *b_mat_h, int *c_mat_h, int i, int j, int k) {
  for (int row = 0; row < i; row++) {
    for (int col = 0; col < k; col++) {
      int val = 0;
      for (int m = 0; m < j; m++) {
        val += a_mat_h[m + row * j] * b_mat_h[col + m * k];
      }
      c_mat_h[col + row * k] = val;
    }
  }
}

int main() {
  int x, y, z;
  std::cout << "Enter matrix dim: ";
  std::cin >> x >> y >> z;
  int *a_mat_h = new int[x * y];
  int *b_mat_h = new int[y * z];
  int *c_mat_h = new int[x * z];
  int *c_mat_h_c = new int[x * z];

  for (int i = 0; i < x; i++)
    for (int j = 0; j < y; j++)
      a_mat_h[i * y + j] = 1;
  for (int i = 0; i < y; i++)
    for (int j = 0; j < z; j++)
      b_mat_h[i * z + j] = 1;

  matmul(a_mat_h, b_mat_h, c_mat_h, x, y, z);
  matmul_cpu(a_mat_h, b_mat_h, c_mat_h_c, x, y, z);

  for (int i = 0; i < x * z; ++i)
    if (c_mat_h[i] != c_mat_h_c[i]) {
      std::cout << "Found unequal at i: " << i << ": " << c_mat_h[i] << ", "
                << c_mat_h_c[i] << std::endl;
      goto Here;
    }
  std::cout << "Success" << std::endl;

Here:
  delete[] a_mat_h;
  delete[] b_mat_h;
  delete[] c_mat_h;

  return 0;
}
