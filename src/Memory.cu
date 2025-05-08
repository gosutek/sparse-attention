#include <cstdio>

#include "Utils.cpp"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

// TODO: Read binary file size
// TODO: Decide on how to pass the input, filename

int main()
{
	void*  host_ptr = nullptr;
	size_t size = 0;
	CUDA_CHECK(cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped));
}
