#include "softmax.cuh"

__global__ void softmax(
	const float* __restrict__ a,
	const size_t m,
	const size_t k,
	float*       acc,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	// TODO: std::expf()
	float e = std::exp(get_elem_rm(a, k, y, x));
	atomicAdd(acc, e);

	__syncthreads();

	float val = e / *acc;
	set_elem_rm(res, k, y, x, val);
}
