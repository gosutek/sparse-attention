#include "spmm.cuh"

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK                     |
  * +------------------------------------------------------------------------------+
*/

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

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM                      |
  * +------------------------------------------------------------------------------+
*/

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

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_NNZWISE_COALESCED                        |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_coalesced_nnzwise(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	extern __shared__ float smem[];
	float*                  dn_row_smem = smem;
	float*                  warp_sums = smem + k;  // INFO: This is aligned to float (fosu)

	for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		dn_row_smem[i] = get_elem_rm(dn, k, blockIdx.y, i);
	}
	__syncthreads();

	float acc = 0.0f;
	for (uint32_t i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += dn_row_smem[row_idx[i]] * val[i];
	}
	__syncwarp();

	for (uint32_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}
	// "acc" now contains the sum across all threads of the warp

	uint32_t lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	uint32_t warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// "warp_sums" now contains the sums across all warps

	__syncthreads();

	const uint32_t warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		float acc = warp_sums[lane_id];

		const uint32_t mask = LOWER_BITS_MASK(warp_cnt);

		for (uint32_t i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
		}

		if (lane_id == 0) {
			set_elem_rm(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

__global__ void _k_ispmm_coalesced_nnzwise(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	extern __shared__ float smem[];
	float*                  dn_row_smem = smem;
	float*                  warp_sums = smem + k;  // INFO: This is aligned to float (fosu)

	for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		dn_row_smem[i] = get_elem_rm(dn, k, blockIdx.y, i);
	}
	__syncthreads();

	float acc = 0.0f;
	for (uint32_t i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += dn_row_smem[row_idx[i]] * val[i];
	}
	__syncwarp();

	for (uint32_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}
	// "acc" now contains the sum across all threads of the warp

	uint32_t lane_id = threadIdx.x & (_CONSTANTS_WARP_SIZE - 1);
	uint32_t warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// "warp_sums" now contains the sums across all warps

	__syncthreads();

	const uint32_t warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		float acc = warp_sums[lane_id];

		const uint32_t mask = LOWER_BITS_MASK(warp_cnt);

		for (uint32_t i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
		}

		if (lane_id == 0) {
			set_elem_rm(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

/*
  * +------------------------------------------------------------------------------+
  * |                SPMM_KERNEL_TYPE_NNZWISE_COALESCED_NO_SMEM                    |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_ispmm_coalesced_nnzwise_no_smem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	float acc = 0.0f;
	for (size_t i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += get_elem_rm(dn, k, blockIdx.y, row_idx[i]) * val[i];
	}
	__syncwarp();

	for (size_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const uint32_t lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const uint32_t warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	const uint32_t          warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	extern __shared__ float warp_sums[];

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}

	__syncthreads();

	if (warp_id == 0) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		float acc = warp_sums[lane_id];

		const uint32_t mask = LOWER_BITS_MASK(warp_cnt);

		for (size_t i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
		}

		if (lane_id == 0) {
			set_elem_rm(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}
