#include "spmm.cuh"

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK                     |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_naive_elemwise_gmem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	const u32 x = blockIdx.x * blockDim.x + threadIdx.x;  // col of res
	const u32 y = blockIdx.y * blockDim.y + threadIdx.y;  // row of res

	f32 acc = 0.0f;
	for (u32 i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
		acc += _d_dn_cm_get(dn, k, col_idx[i], x) * val[i];
	}
	_d_dn_rm_set(res, n, y, x, acc);
}

__global__ void _k_ispmm_naive_elemwise_gmem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

	f32 acc = 0.0f;
	for (u32 i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {      // 1 LDG
		acc += _d_dn_rm_get(dn, k, y, row_idx[i]) * val[i];  // 2 LDG
	}
	_d_dn_rm_set(res, n, y, x, acc);
}

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM                      |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_naive_elemwise_smem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	const u32 r = threadIdx.x;
	const u32 c = blockIdx.x;

	f32 acc = 0.0f;

	extern __shared__ f32 dn_col_smem[];

	dn_col_smem[r] = _d_dn_cm_get(dn, k, r, c);
	__syncthreads();

	for (u32 i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
		acc += dn_col_smem[col_idx[i]] * val[i];
	}

	_d_dn_rm_set(res, n, r, c, acc);
}

__global__ void _k_ispmm_naive_elemwise_smem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	const u32 c = threadIdx.x;
	const u32 r = blockIdx.x;

	f32 acc = 0.0f;

	extern __shared__ f32 dn_row_smem[];

	dn_row_smem[c] = _d_dn_rm_get(dn, k, r, c);
	__syncthreads();

	for (u32 i = col_ptr[c]; i < col_ptr[c + 1]; ++i) {
		acc += dn_row_smem[row_idx[i]] * val[i];
	}

	_d_dn_rm_set(res, n, r, c, acc);
}

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_NNZWISE_COALESCED                        |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_coalesced_nnzwise(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	extern __shared__ f32 smem[];
	f32* const            dn_col_smem = smem;
	f32* const            warp_sums = smem + k;  // INFO: This is aligned to f32 (fosu)

	for (u32 i = threadIdx.x; i < k; i += blockDim.x) {
		dn_col_smem[i] = _d_dn_cm_get(dn, k, i, blockIdx.x);
	}
	__syncthreads();

	f32 acc = 0.0f;
	for (u32 i = row_ptr[blockIdx.y] + threadIdx.x; i < row_ptr[blockIdx.y + 1]; i += blockDim.x) {
		acc += dn_col_smem[col_idx[i]] * val[i];
	}
	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}
	u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// "warp_sums" now contains the sums across all warps

	__syncthreads();

	const u32 warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		f32 acc = warp_sums[lane_id];

		const u32 mask = LOWER_BITS_MASK(warp_cnt);

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}

		if (lane_id == 0) {
			_d_dn_rm_set(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

__global__ void _k_ispmm_coalesced_nnzwise(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	extern __shared__ f32 smem[];
	f32* const            dn_row_smem = smem;
	f32* const            warp_sums = smem + k;  // INFO: This is aligned to f32 (fosu)

	for (u32 i = threadIdx.x; i < k; i += blockDim.x) {
		dn_row_smem[i] = _d_dn_rm_get(dn, k, blockIdx.y, i);
	}
	__syncthreads();

	f32 acc = 0.0f;
	for (u32 i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += dn_row_smem[row_idx[i]] * val[i];
	}
	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}
	// "acc" now contains the sum across all threads of the warp

	u32 lane_id = threadIdx.x & (_CONSTANTS_WARP_SIZE - 1);
	u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// "warp_sums" now contains the sums across all warps

	__syncthreads();

	const u32 warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		f32 acc = warp_sums[lane_id];

		const u32 mask = LOWER_BITS_MASK(warp_cnt);

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}

		if (lane_id == 0) {
			_d_dn_rm_set(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

/*
  * +------------------------------------------------------------------------------+
  * |                SPMM_KERNEL_TYPE_NNZWISE_COALESCED_NO_SMEM                    |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_coalesced_nnzwise_no_smem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	f32 acc = 0.0f;
	for (u32 i = row_ptr[blockIdx.y] + threadIdx.x; i < row_ptr[blockIdx.y + 1]; i += blockDim.x) {
		acc += _d_dn_cm_get(dn, k, col_idx[i], blockIdx.x) * val[i];
	}
	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	const u32             warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	extern __shared__ f32 warp_sums[];

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}

	__syncthreads();

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		f32 acc = warp_sums[lane_id];

		const u32 mask = LOWER_BITS_MASK(warp_cnt);

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
		}

		if (lane_id == 0) {
			_d_dn_rm_set(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

__global__ void _k_ispmm_coalesced_nnzwise_no_smem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	f32 acc = 0.0f;
	for (u32 i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += _d_dn_rm_get(dn, k, blockIdx.y, row_idx[i]) * val[i];
	}
	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

	const u32             warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	extern __shared__ f32 warp_sums[];

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}

	__syncthreads();

	if (warp_id == 0 && lane_id < warp_cnt) {
		// WARN: some threads point to garbage, should be fine as they don't contribute due to 'mask'
		f32 acc = warp_sums[lane_id];

		const u32 mask = LOWER_BITS_MASK(warp_cnt);

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
		}

		if (lane_id == 0) {
			_d_dn_rm_set(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

/*
  * +------------------------------------------------------------------------------+
  * |                    SPMM_KERNEL_TYPE_NNZWISE_VECTORIZED                       |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_vectorized_nnzwise_regs(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	constexpr u32 TK = 4;  // non-zeros assigned for each *thread*

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

	const u32 base_unaligned_i = row_ptr[blockIdx.y];  // 2 LDG (but hardware performs a single load)

	// only blockIdx.z = 1 takes account of the rem elements in blockIdx.z = 0
	// NOTE: You can do a branch here, it doesn't diverge any threads since its across the z dimension of the grid, i.e. all warps of a block will enter the same branch
	u32 ri_aligned_i = base_unaligned_i;  // row_idx aligned index
	while (!is_aligned(&col_idx[ri_aligned_i], 16)) {
		++ri_aligned_i;
	}
	const u32 ri_unaligned_cnt = ri_aligned_i - base_unaligned_i;  // exclusive of block_aligned_start, because it will be vectorized

	u32 v_aligned_i = base_unaligned_i;  // val aligned index
	while (!is_aligned(&val[v_aligned_i], 16)) {
		++v_aligned_i;
	}

	const u32 row_end_i = row_ptr[blockIdx.y + 1];

	const u32 row_nnz = row_end_i - ri_aligned_i;  // count[ri_aligned_i - col_end_i) (261)
	static_assert(TK % 2 == 0);
	const u32 n_tail_loads = MOD_POW2(row_nnz, TK);
	const u32 n_velems = row_nnz - n_tail_loads;      // 261 - 1 = 260
	const u32 n_vloads = n_velems / TK;               // (260 / 4 = 65)
	const u32 n_vloads_block = n_vloads / gridDim.z;  // (65 // 2 = 32)
	const u32 rem_n_vloads = n_vloads % gridDim.z;    // (65 % 2 = 1)
	const u32 nnz_block = n_velems / gridDim.z;       // 260 / 2 = 130

	// assert(blockDim.y >= n_vloads_block);  // at least as many threads per block as vectorized loads per block

	// for the first warp we should split its first lane to take care of the unaligned elements
	// while the rest of the lanes tackle the vectorized loads

	const u32         lane = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const u32         warp = threadIdx.x / _CONSTANTS_WARP_SIZE;
	f32               acc = 0.0f;
	__align__(16) u32 t_col_idx[TK] = { 0 };
	__align__(16) f32 t_val[TK] = { 0.0f };

	if (blockIdx.z == 0 && warp == 0 && lane == 0) {  // WD
		for (u32 i = 0; i < ri_unaligned_cnt; ++i) {  // up to 3 iterations
			acc += _d_dn_cm_get(dn, k, col_idx[base_unaligned_i + i], blockIdx.x) * val[base_unaligned_i + i];
		}
		for (u32 i = 0; i < n_tail_loads; ++i) {  // up to 3 iterations
			acc += _d_dn_cm_get(dn, k, col_idx[ri_aligned_i + gridDim.z * nnz_block + i], blockIdx.x) * val[ri_aligned_i + gridDim.z * nnz_block + i];
		}
	}
	u32 block_start = 0;
	u32 block_end = 0;
	if (blockIdx.z == 0) {
		block_start = ri_aligned_i;
		block_end = block_start + (n_vloads_block + rem_n_vloads) * TK;
	} else {
		block_start = ri_aligned_i + (blockIdx.z * n_vloads_block + rem_n_vloads) * TK;
		block_end = block_start + n_vloads_block * TK;
	}

	for (u32 i = block_start + threadIdx.x * TK; i < block_end; i += blockDim.x * TK) {
		const uint4* __restrict__ row_idx_v = reinterpret_cast<const uint4*>(__builtin_assume_aligned(&col_idx[i], 16));
		const float4* __restrict__ val_v = reinterpret_cast<const float4*>(__builtin_assume_aligned(&val[i], 16));
		reinterpret_cast<uint4*>(&t_col_idx)[0] = row_idx_v[0];
		reinterpret_cast<float4*>(&t_val)[0] = val_v[0];

		acc += _d_dn_cm_get(dn, k, t_col_idx[0], blockIdx.y) * t_val[0];
		acc += _d_dn_cm_get(dn, k, t_col_idx[1], blockIdx.y) * t_val[1];
		acc += _d_dn_cm_get(dn, k, t_col_idx[2], blockIdx.y) * t_val[2];
		acc += _d_dn_cm_get(dn, k, t_col_idx[3], blockIdx.y) * t_val[3];
	}

	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const u32 warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	// extern __shared__ f32   warp_sums[n_warps];
	extern __shared__ f32 warp_sums[];

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
		f32 acc = warp_sums[lane];

		constexpr u32 mask = 0x3;

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}
		if (lane == 0) {
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

__global__ void _k_ispmm_vectorized_nnzwise_regs(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res)
{
	constexpr u32 TK = 4;  // non-zeros assigned for each *thread*

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

	const u32 base_unaligned_i = col_ptr[blockIdx.x];  // 2 LDG (but hardware performs a single load)

	// only blockIdx.z = 1 takes account of the rem elements in blockIdx.z = 0
	// NOTE: You can do a branch here, it doesn't diverge any threads since its across the z dimension of the grid, i.e. all warps of a block will enter the same branch
	u32 ri_aligned_i = base_unaligned_i;  // row_idx aligned index
	while (!is_aligned(&row_idx[ri_aligned_i], 16)) {
		++ri_aligned_i;
	}
	const u32 ri_unaligned_cnt = ri_aligned_i - base_unaligned_i;  // exclusive of block_aligned_start, because it will be vectorized

	u32 v_aligned_i = base_unaligned_i;  // val aligned index
	while (!is_aligned(&val[v_aligned_i], 16)) {
		++v_aligned_i;
	}

	const u32 col_end_i = col_ptr[blockIdx.x + 1];

	const u32 col_nnz = col_end_i - ri_aligned_i;     // count[ri_aligned_i - col_end_i) (261)
	const u32 n_tail_loads = col_nnz & (TK - 1);      // ( 261 % 4 = 1)
	const u32 n_velems = col_nnz - n_tail_loads;      // 261 - 1 = 260
	const u32 n_vloads = n_velems / TK;               // (260 / 4 = 65)
	const u32 n_vloads_block = n_vloads / gridDim.z;  // (65 // 2 = 32)
	const u32 rem_n_vloads = n_vloads % gridDim.z;    // (65 % 2 = 1)
	const u32 nnz_block = n_velems / gridDim.z;       // 260 / 2 = 130

	// assert(blockDim.x >= n_vloads_block);  // at least as many threads per block as vectorized loads per block

	// for the first warp we should split its first lane to take care of the unaligned elements
	// while the rest of the lanes tackle the vectorized loads

	const u32         lane = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
	const u32         warp = threadIdx.x / _CONSTANTS_WARP_SIZE;
	f32               acc = 0.0f;
	__align__(16) u32 t_row_idx[TK] = { 0 };
	__align__(16) f32 t_val[TK] = { 0.0f };

	if (blockIdx.z == 0 && warp == 0 && lane == 0) {  // WD
		for (u32 i = 0; i < ri_unaligned_cnt; ++i) {  // up to 3 iterations
			acc += _d_dn_rm_get(dn, k, blockIdx.y, row_idx[base_unaligned_i + i]) * val[base_unaligned_i + i];
			// acc += x_row_smem[row_idx[base_unaligned_i + i]] * val[base_unaligned_i + i];
		}
		for (u32 i = 0; i < n_tail_loads; ++i) {  // up to 3 iterations
			acc += _d_dn_rm_get(dn, k, blockIdx.y, row_idx[ri_aligned_i + gridDim.z * nnz_block + i]) * val[ri_aligned_i + gridDim.z * nnz_block + i];
			// acc += x_row_smem[row_idx[ri_aligned_i + gridDim.z * nnz_block + i]] * val[ri_aligned_i + gridDim.z * nnz_block + i];
		}
	}
	u32 block_start = 0;
	u32 block_end = 0;
	if (blockIdx.z == 0) {
		block_start = ri_aligned_i;
		block_end = block_start + (n_vloads_block + rem_n_vloads) * TK;
	} else {
		block_start = ri_aligned_i + (blockIdx.z * n_vloads_block + rem_n_vloads) * TK;
		block_end = block_start + n_vloads_block * TK;
	}

	for (u32 i = block_start + threadIdx.x * TK; i < block_end; i += blockDim.x * TK) {
		const uint4* __restrict__ row_idx_v = reinterpret_cast<const uint4*>(__builtin_assume_aligned(&row_idx[i], 16));
		const float4* __restrict__ val_v = reinterpret_cast<const float4*>(__builtin_assume_aligned(&val[i], 16));
		reinterpret_cast<uint4*>(&t_row_idx)[0] = row_idx_v[0];
		reinterpret_cast<float4*>(&t_val)[0] = val_v[0];

		acc += _d_dn_rm_get(dn, k, blockIdx.y, t_row_idx[0]) * t_val[0];
		acc += _d_dn_rm_get(dn, k, blockIdx.y, t_row_idx[1]) * t_val[1];
		acc += _d_dn_rm_get(dn, k, blockIdx.y, t_row_idx[2]) * t_val[2];
		acc += _d_dn_rm_get(dn, k, blockIdx.y, t_row_idx[3]) * t_val[3];
	}

	__syncwarp();

	for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
	}

	const u32 warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
	// extern __shared__ f32   warp_sums[n_warps];
	extern __shared__ f32 warp_sums[];

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
		f32 acc = warp_sums[lane];

		constexpr u32 mask = 0x3;

		for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, warp_cnt);
		}
		if (lane == 0) {
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

/*
  * +------------------------------------------------------------------------------+
  * |                   SPMM_KERNEL_TYPE_NNZWISE_COLUMN_TILING                     |
  * +------------------------------------------------------------------------------+
*/

__global__ void _k_spmm_column_tiling_nnzwise(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	const u32 bm,
	f32* __restrict__ res)
{
	for (u32 r = 0; r < bm; ++r) {
		f32 acc = 0.0f;
		for (u32 i = row_ptr[blockIdx.x * bm + r] + threadIdx.x; i < row_ptr[blockIdx.x * bm + r + 1]; i += blockDim.x) {
			acc += _d_dn_cm_get(dn, k, col_idx[i], blockIdx.y) * val[i];
		}
		__syncwarp();

		for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
		}

		const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
		const u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

		const u32             warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
		extern __shared__ f32 warp_sums[];

		if (lane_id == 0) {
			warp_sums[warp_id] = acc;
		}

		__syncthreads();

		if (warp_id == 0 && lane_id < warp_cnt) {
			// WARN: some threads point to garbage
			f32 acc = warp_sums[lane_id];

			const u32 mask = LOWER_BITS_MASK(warp_cnt);

			for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
				acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
			}

			if (lane_id == 0) {
				_d_dn_rm_set(res, n, blockIdx.x * bm + r, blockIdx.y, acc);
			}
		}
	}
}

__global__ void _k_ispmm_column_tiling_nnzwise(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	const u32 bn,
	f32* __restrict__ res)
{
	for (u32 c = 0; c < bn; ++c) {
		f32 acc = 0.0f;
		for (u32 i = col_ptr[blockIdx.x * bn + c] + threadIdx.x; i < col_ptr[blockIdx.x * bn + c + 1]; i += blockDim.x) {
			acc += _d_dn_rm_get(dn, k, blockIdx.y, row_idx[i]) * val[i];
		}
		__syncwarp();

		for (u32 i = _CONSTANTS_WARP_SIZE / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(0xffffffff, acc, i, _CONSTANTS_WARP_SIZE);
		}

		const u32 lane_id = MOD_POW2(threadIdx.x, _CONSTANTS_WARP_SIZE);
		const u32 warp_id = threadIdx.x / _CONSTANTS_WARP_SIZE;

		const u32             warp_cnt = blockDim.x / _CONSTANTS_WARP_SIZE;
		extern __shared__ f32 warp_sums[];

		if (lane_id == 0) {
			warp_sums[warp_id] = acc;
		}

		__syncthreads();

		if (warp_id == 0 && lane_id < warp_cnt) {
			// WARN: some threads point to garbage
			f32 acc = warp_sums[lane_id];

			const u32 mask = LOWER_BITS_MASK(warp_cnt);

			for (u32 i = warp_cnt / 2; i > 0; i /= 2) {
				acc += __shfl_xor_sync(mask, acc, i, _CONSTANTS_WARP_SIZE);
			}

			if (lane_id == 0) {
				_d_dn_rm_set(res, n, blockIdx.y, blockIdx.x * bn + c, acc);
			}
		}
	}
}
