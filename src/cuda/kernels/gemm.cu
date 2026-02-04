#include "gemm.cuh"

__global__ void gemm(
	const float* __restrict__ a,  // row-major
	const float* __restrict__ b,  // col-major
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = threadIdx.x;
	uint32_t y = blockIdx.x;

	if (x >= n || y >= m) {  // not really needed
		return;
	}

	float acc = 0.0f;
	// TODO: Change hardcoded value
	__shared__ float a_row_sm[512];

	a_row_sm[x] = get_elem_rm(a, k, y, x);
	__syncthreads();

	for (size_t i = 0; i < k; ++i) {
		acc += a_row_sm[i] * b[x * k + i];
	}
	set_elem_rm(res, n, y, x, acc);
}
