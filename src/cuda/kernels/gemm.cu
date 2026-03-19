#include "cuda_helpers.cuh"
#include "helpers.h"

__global__ void gemm(
	const f32* __restrict__ a,  // row-major
	const f32* __restrict__ b,  // col-major
	const size_t m,
	const size_t k,
	const size_t n,
	f32* __restrict__ res)
{
	u32 x = threadIdx.x;
	u32 y = blockIdx.x;

	if (x >= n || y >= m) {  // not really needed
		return;
	}

	f32 acc = 0.0f;
	// TODO: Change hardcoded value
	__shared__ f32 a_row_sm[512];

	a_row_sm[x] = _d_dn_rm_get(a, k, y, x);
	__syncthreads();

	for (size_t i = 0; i < k; ++i) {
		acc += a_row_sm[i] * b[x * k + i];
	}
	_d_dn_rm_set(res, n, y, x, acc);
}
