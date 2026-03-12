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
	float*                  dn_col_smem = smem;
	float*                  warp_sums = smem + k;  // INFO: This is aligned to float (fosu)

	for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		dn_col_smem[i] = get_elem_cm(dn, k, i, blockIdx.x);
	}
	__syncthreads();

	float acc = 0.0f;
	for (uint32_t i = row_ptr[blockIdx.y] + threadIdx.x; i < row_ptr[blockIdx.y + 1]; i += blockDim.x) {
		acc += dn_col_smem[col_idx[i]] * val[i];
	}
	__syncwarp();

	for (uint32_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}
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
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
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
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
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

__global__ void _k_spmm_coalesced_nnzwise_no_smem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	float acc = 0.0f;
	for (size_t i = row_ptr[blockIdx.y] + threadIdx.x; i < row_ptr[blockIdx.y + 1]; i += blockDim.x) {
		acc += get_elem_cm(dn, k, col_idx[i], blockIdx.x) * val[i];
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

	if (warp_id == 0 && lane_id < warp_cnt) {
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

	if (warp_id == 0 && lane_id < warp_cnt) {
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

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_NNZWISE_VECTORIZED                       |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_ispmm_vectorized_nnzwise_regs(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	constexpr size_t TK = 4;  // non-zeros assigned for each *thread*

	// __shared__ float x_row_smem[MAT_SIZE];

	// NOTE: Coalesced acccess, plain
	// for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
	// 	x_row_sm[i] = get_elem_rm(a, k, blockIdx.y, i);
	// }

	// NOTE: Coalesced access + Vectorized loads
	// for (size_t i = threadIdx.x * TK; i < MAT_SIZE; i += blockDim.x * TK) {
	// 	const float4 f4 = reinterpret_cast<const float4*>(&a[blockIdx.y * k + i])[0];
	//
	// 	x_row_smem[i] = f4.x;
	// 	x_row_smem[i + 1] = f4.y;
	// 	x_row_smem[i + 2] = f4.z;
	// 	x_row_smem[i + 3] = f4.w;
	//
	// 	// NOTE: this loop doesn't get vectorized for some reason
	// 	// #pragma unroll
	// 	// 		for (size_t t = 0; t < TK; ++t) {
	// 	// 			// x_row_sm[i + t] = reinterpret_cast<const float*>(&tmp)[0];
	// 	// 			// x_row_sm[i + t] = ((float*)&tmp)[0];
	// 	// 		}
	// }

	// __syncthreads();

	/*
      * +-------------------------------------------------ROW_IDX------------------------------------------------+
      *           row_idx_gmem_aligned_start
      *                      |        + blockDim.z * block_nnz
      *                      v                     v
      * +--------------------+---------------------+---------------------+-----+---------------------+-----------+
      * | unaligned elements | vectorized elements | vectorized elements | ... | vectorized elements | remainder |
      * +--------------------+---------------------+---------------------+-----+---------------------+-----------+
      * +--------------blockIdx.z = 0--------------+----blockIdx.z = 1---+-...-+----blockIdx.z = gridDim.z - 1---+
      * +-----------------block_nnz----------------+------block_nnz------+-----+------block_nnz------+-block_rem-+
   */

	const size_t base_unaligned_i = col_ptr[blockIdx.x];  // 2 LDG (but hardware performs a single load)

	// only blockIdx.z = 1 takes account of the rem elements in blockIdx.z = 0
	// NOTE: You can do a branch here, it doesn't diverge any threads since its across the z dimension of the grid, i.e. all warps of a block will enter the same branch
	size_t ri_aligned_i = base_unaligned_i;  // row_idx aligned index
	while (!is_aligned(&row_idx[ri_aligned_i], 16)) {
		++ri_aligned_i;
	}
	const size_t ri_unaligned_cnt = ri_aligned_i - base_unaligned_i;  // exclusive of block_aligned_start, because it will be vectorized

	size_t v_aligned_i = base_unaligned_i;  // val aligned index
	while (!is_aligned(&val[v_aligned_i], 16)) {
		++v_aligned_i;
	}
	const size_t v_unaligned_cnt = v_aligned_i - base_unaligned_i;  // exclusive of val_gmem_aligned_start_idx, because it will be vectorized
	// assert(ri_unaligned_cnt == v_unaligned_cnt);

	const size_t col_end_i = col_ptr[blockIdx.x + 1];

	const size_t   col_nnz = col_end_i - ri_aligned_i;     // count[ri_aligned_i - col_end_i) (261)
	const uint32_t n_tail_loads = col_nnz & (TK - 1);      // ( 261 % 4 = 1)
	const size_t   n_velems = col_nnz - n_tail_loads;      // 261 - 1 = 260
	const size_t   n_vloads = n_velems / TK;               // (260 / 4 = 65)
	const size_t   n_vloads_block = n_vloads / gridDim.z;  // (65 // 2 = 32)
	const size_t   rem_n_vloads = n_vloads % gridDim.z;    // (65 % 2 = 1)
	const size_t   nnz_block = n_velems / gridDim.z;       // 260 / 2 = 130
	const uint32_t n_scalar_loads = n_tail_loads + ri_unaligned_cnt;

	// assert(blockDim.x >= n_vloads_block);  // at least as many threads per block as vectorized loads per block

	// for the first warp we should split its first lane to take care of the unaligned elements
	// while the rest of the lanes tackle the vectorized loads

	const uint32_t         lane = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const uint32_t         warp = threadIdx.x / _CONSTANTS_WARP_SIZE;
	float                  acc = 0.0f;
	__align__(16) uint32_t t_row_idx[TK] = { 0 };
	__align__(16) float    t_val[TK] = { 0.0f };

	if (blockIdx.z == 0 && warp == 0 && lane == 0) {     // WD
		for (size_t i = 0; i < ri_unaligned_cnt; ++i) {  // up to 3 iterations
			acc += get_elem_rm(dn, k, blockIdx.y, row_idx[base_unaligned_i + i]) * val[base_unaligned_i + i];
			// acc += x_row_smem[row_idx[base_unaligned_i + i]] * val[base_unaligned_i + i];
		}
		for (size_t i = 0; i < n_tail_loads; ++i) {  // up to 3 iterations
			acc += get_elem_rm(dn, k, blockIdx.y, row_idx[ri_aligned_i + gridDim.z * nnz_block + i]) * val[ri_aligned_i + gridDim.z * nnz_block + i];
			// acc += x_row_smem[row_idx[ri_aligned_i + gridDim.z * nnz_block + i]] * val[ri_aligned_i + gridDim.z * nnz_block + i];
		}
	}
	size_t block_start = 0;
	size_t block_end = 0;
	if (blockIdx.z == 0) {
		block_start = ri_aligned_i;
		block_end = block_start + (n_vloads_block + rem_n_vloads) * TK;
	} else {
		block_start = ri_aligned_i + (blockIdx.z * n_vloads_block + rem_n_vloads) * TK;
		block_end = block_start + n_vloads_block * TK;
	}

	for (size_t i = block_start + threadIdx.x * TK; i < block_end; i += blockDim.x * TK) {
		const uint4* __restrict__ row_idx_v = reinterpret_cast<const uint4*>(__builtin_assume_aligned(&row_idx[i], 16));
		const float4* __restrict__ val_v = reinterpret_cast<const float4*>(__builtin_assume_aligned(&val[i], 16));
		reinterpret_cast<uint4*>(&t_row_idx)[0] = row_idx_v[0];
		reinterpret_cast<float4*>(&t_val)[0] = val_v[0];

		acc += get_elem_rm(dn, k, blockIdx.y, t_row_idx[0]) * t_val[0];
		acc += get_elem_rm(dn, k, blockIdx.y, t_row_idx[1]) * t_val[1];
		acc += get_elem_rm(dn, k, blockIdx.y, t_row_idx[2]) * t_val[2];
		acc += get_elem_rm(dn, k, blockIdx.y, t_row_idx[3]) * t_val[3];
	}

	__syncwarp();

	for (size_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const uint32_t warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	// extern __shared__ float   warp_sums[n_warps];
	extern __shared__ float warp_sums[];

	// at this point the first thread (lane_id == 0) of every warp in this block
	// has the result from TK non-zeros for this col
	// this is essentially warp-wide reduction
	if (lane == 0) {
		warp_sums[warp] = acc;
	}
	// we write the warp-wide results to a block-wide memory location (SMEM)
	// so that we can perform block-wide reduction

	__syncthreads();

	// we assign the block-wide reduction to warp-0
	if (warp == 0 && lane < warp_cnt) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane];

		constexpr uint32_t mask = 0x3;

		for (size_t i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}
		if (lane == 0) {
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

// TODO: Give this kernel a proper name
/*
  * +------------------------------------------------------------------------------+
  * |                       SPMM_KERNEL_TYPE_NNZWISE_FINAL                         |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_ispmm_coalesced_nnzwise_last(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	const size_t bn,
	float* __restrict__ res)
{
	for (size_t c = 0; c < bn; ++c) {
		float acc = 0.0f;
		for (size_t i = col_ptr[blockIdx.x * bn + c] + threadIdx.x; i < col_ptr[blockIdx.x * bn + c + 1]; i += blockDim.x) {
			acc += get_elem_rm(dn, k, blockIdx.y, row_idx[i]) * val[i];
		}
		__syncwarp();

		for (size_t i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
		}

		const uint32_t lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
		const uint32_t warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

		const uint32_t n_warps = blockDim.x / _CONSTANTS_WARP_SIZE;
		// extern __shared__ float   warp_sums[n_warps];
		extern __shared__ float warp_sums[];

		if (lane_id == 0) {
			warp_sums[warp_id] = acc;
		}

		__syncthreads();

		if (warp_id == 0) {
			// WARN: some threads point to garbage
			float acc = warp_sums[lane_id];

			constexpr uint32_t mask = 0xFF;

			for (size_t i = n_warps / 2; i > 0; i /= 2) {
				acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
			}

			if (lane_id == 0) {
				set_elem_rm(res, n, blockIdx.y, blockIdx.x * bn + c, acc);
			}
		}
	}
}
