#include "cu_helpers.cuh"
#include "helpers.h"

__global__ void softmax(
	const f32* __restrict__ a,
	const size_t m,
	const size_t k,
	f32*         acc,
	f32* __restrict__ res)
{
	u32 x = blockIdx.x * blockDim.x + threadIdx.x;
	u32 y = blockIdx.y * blockDim.y + threadIdx.y;

	// TODO: std::expf()
	f32 e = std::exp(_d_dn_rm_get(a, k, y, x));
	atomicAdd(acc, e);

	__syncthreads();

	f32 val = e / *acc;
	_d_dn_rm_set(res, k, y, x, val);
}
