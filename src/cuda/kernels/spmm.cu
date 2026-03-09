#include "spmm.cuh"
#include <sys/types.h>

__global__ void _k_spmm_naive_elemwise_gmem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,  // sparse rows
	const uint32_t k,  // sparse cols
	const uint32_t n,  // dense cols
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

// __global__ void _k_spmm_naive_elemwise_smem(
// 	const uint32_t* __restrict__ row_ptr,
// 	const uint32_t* __restrict__ col_idx,
// 	const float* __restrict__ val,
// 	const float* __restrict__ dn,
// 	const uint32_t m,
// 	const uint32_t k,
// 	const uint32_t n,
// 	float* __restrict__ res)
// {
// 	uint32_t x = threadIdx.x;
// 	uint32_t y = blockIdx.x;
//
// 	float acc = 0.0f;
//
// 	extern __shared__ float dn_col_smem[];
//
// 	dn_col_smem[x] = get_elem_rm(a, k, y, x);
// 	__syncthreads();
//
//   for (uint32_t i = row_ptr[])
// 	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
// 		acc += x_row_smem[row_idx[i]] * val[i];
// 	}
//
// 	set_elem_rm(res, n, y, x, acc);
// }
//
// __global__ void _k_ispmm_naive_elemwise_smem(
// 	const float* __restrict__ dn,
// 	const uint32_t* __restrict__ col_ptr,
// 	const uint32_t* __restrict__ row_idx,
// 	const float* __restrict__ val,
// 	const uint32_t m,
// 	const uint32_t k,
// 	const uint32_t n,
// 	float* __restrict__ res)
// {
// 	uint32_t x = threadIdx.x;
// 	uint32_t y = blockIdx.x;
//
// 	float acc = 0.0f;
//
// 	extern __shared__ float x_row_smem[];
//
// 	x_row_smem[x] = get_elem_rm(dn, k, y, x);
// 	__syncthreads();
//
// 	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
// 		acc += x_row_smem[row_idx[i]] * val[i];
// 	}
//
// 	set_elem_rm(res, n, y, x, acc);
// }
