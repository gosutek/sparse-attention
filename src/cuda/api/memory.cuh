#pragma once

#include <stdio.h>

// TODO: This *can* leak memory
#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

void* cuda_malloc_device(size_t b_size);
void* cuda_malloc_host(size_t b_size);
void  cuda_dealloc_host(void* ptr);
void  cuda_dealloc_device(void* ptr);
