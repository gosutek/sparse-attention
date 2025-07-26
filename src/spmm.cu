#include "common.h"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

[[maybe_unused]] static void* malloc_device(size_t size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, size));
	return ptr;
}

void* malloc_host(size_t size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMallocHost(&ptr, size));
	return ptr;
}

void dealloc_host(void* ptr)
{
	cudaFreeHost(ptr);
}
