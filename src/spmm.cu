#include "common.h"
#include "matrix.h"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

[[maybe_unused]] static void* cuda_malloc_device(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, b_size));
	return ptr;
}

void* cuda_device_copy(void* host, size_t b_size)
{
	void* device = nullptr;
	CUDA_CHECK(cudaMalloc(&device, b_size));
	CUDA_CHECK(cudaMemcpy(device, host, b_size, cudaMemcpyHostToDevice));
	return device;
}

void* cuda_malloc_host(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMallocHost(&ptr, b_size));
	return ptr;
}

void cuda_dealloc_host(void* ptr)
{
	cudaFreeHost(ptr);
}

void cuda_dealloc_device(void* ptr)
{
	cudaFree(ptr);
}

void* prepare(Input& input)
{
	// TODO: Streams go here
	void* dev = cuda_device_copy(input.data, input.b_size);
	return dev;
}
