#include "spmm.cuh"
#include <sys/types.h>

__global__ void _k_spmm_naive_elemwise_gmem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;  // col of res
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;  // row of res

	float acc = 0.0f;
	for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
		acc += get_elem_cm(dn, k, col_idx[i], x) * val[i];
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void _k_ispmm_naive_elemwise_gmem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float acc = 0.0f;
	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {  // 1 LDG
		acc += get_elem_rm(dn, k, y, row_idx[i]) * val[i];  // 2 LDG
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void _k_spmm_naive_elemwise_smem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	uint32_t r = threadIdx.x;
	uint32_t c = blockIdx.x;

	float acc = 0.0f;

	extern __shared__ float dn_col_smem[];

	dn_col_smem[r] = get_elem_cm(dn, k, r, c);
	__syncthreads();

	for (uint32_t i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
		acc += dn_col_smem[col_idx[i]] * val[i];
	}

	set_elem_rm(res, n, r, c, acc);
}

__global__ void _k_ispmm_naive_elemwise_smem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	uint32_t c = threadIdx.x;
	uint32_t r = blockIdx.x;

	float acc = 0.0f;

	extern __shared__ float dn_row_smem[];

	dn_row_smem[c] = get_elem_rm(dn, k, r, c);
	__syncthreads();

	for (size_t i = col_ptr[c]; i < col_ptr[c + 1]; ++i) {
		acc += dn_row_smem[row_idx[i]] * val[i];
	}

	set_elem_rm(res, n, r, c, acc);
}
