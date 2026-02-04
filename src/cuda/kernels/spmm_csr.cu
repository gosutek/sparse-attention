#include "spmm_csr.cuh"

__global__ void spmm_naive_elemwise_gmem_csr(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ d,
	const size_t m,  // sparse rows
	const size_t k,  // sparse cols
	const size_t n,  // dense cols
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;  // col of res
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;  // row of res

	float acc = 0.0f;
	for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
		acc += get_elem_cm(d, k, col_idx[i], x) * val[i];
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void spmm_coalesced_elemwise_csr(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = threadIdx.x;
	uint32_t y = blockIdx.x;

	__shared__ float x_row_smem[MAT_SIZE];
	__shared__ float shared_acc[MAT_SIZE];

	for (size_t i = x; i < k; i += blockDim.x) {
		x_row_smem[i] = get_elem_rm(a, k, y, i);
		shared_acc[i] = 0.0f;
	}
	__syncthreads();

	for (size_t row = 0; row < k; ++row) {
		for (size_t i = row_ptr[row] + x; i < row_ptr[row + 1]; i += blockDim.x) {
			atomicAdd_block(&shared_acc[col_idx[i]], x_row_smem[row] * val[i]);
		}
	}

	__syncthreads();

	for (size_t i = x; i < k; i += blockDim.x) {
		set_elem_rm(res, n, y, i, shared_acc[i]);
	}
}
