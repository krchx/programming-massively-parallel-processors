#pragma once

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 32
#define COARS_FACT 4

// Kogge-Stone scan algorithm
void Kogge_Stone_Scan(double *in_h, double *out_h, int n);

// Brent-Kung scan algorithm
void Brent_Kung_Scan(double *in_h, double *out_h, int n);

// Coarsing with Kogge-Stone scan
void Coarsed_Kogge_Stone_Scan(double *in_h, double *out_h, int n);

// Single-pass scan for memory access efficiency
void Single_Pass_Scan(double *in_h, double *out_h, int n);
