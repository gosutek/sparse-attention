#include "memory.cuh"

namespace spmm
{
	void* cuda_malloc_device(size_t b_size)
	{
		void* ptr = nullptr;
		CUDA_CHECK(cudaMalloc(&ptr, b_size));
		return ptr;
	}

	void* cuda_malloc_host(size_t b_size)
	{
		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocHost(&ptr, b_size));
		return ptr;
	}

	void cuda_dealloc_host(void* ptr)
	{
		CUDA_CHECK(cudaFreeHost(ptr));
	}

	void cuda_dealloc_device(void* ptr)
	{
		CUDA_CHECK(cudaFree(ptr));
	}

}  // namespace spmm
