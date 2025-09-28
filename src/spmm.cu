#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cusparse.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "handle.h"
#include "matrix.h"
#include "spmm.cuh"
#include "utils.h"

void* cuda_malloc_device(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, b_size));
	return ptr;
}

void* cuda_malloc_host(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMallocHost(&ptr, b_size));
	return ptr;
}

void cuda_dealloc_host(void* ptr)
{
	CUDA_CHECK(cudaFreeHost(ptr));
}

void cuda_dealloc_device(void* ptr)
{
	CUDA_CHECK(cudaFree(ptr));
}

__device__ inline static bool is_aligned(const void* addr, const size_t alignment_bytes)
{
	return (reinterpret_cast<uintptr_t>(addr) & (alignment_bytes - 1)) == 0;
}

/*
 * This aligns relative to @param base
 */
// NOTE: is relative align necessary?
__device__ inline static uintptr_t align(const void* base, const void* addr, const size_t alignment_bytes)
{
	const uintptr_t offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(base);
	const uintptr_t aligned_offset = (reinterpret_cast<uintptr_t>(offset) + (alignment_bytes - 1)) & ~size_t(alignment_bytes - 1);
	return reinterpret_cast<uintptr_t>(base) + aligned_offset;
}

__device__ inline static float get_elem_rm(const float* const a, size_t n_cols, size_t row, size_t col)
{
	return a[row * n_cols + col];
}

[[maybe_unused]] __device__ inline static float get_elem_cm(const float* const a, size_t n_rows, size_t row, size_t col)
{
	return a[col * n_rows + row];
}

__device__ inline static void set_elem_rm(float* const a, size_t n_cols, size_t row, size_t col, float val)
{
	a[row * n_cols + col] = val;
}

__device__ inline static void set_elem_cm(float* const a, size_t n_rows, size_t row, size_t col, float val)
{
	a[col * n_rows + row] = val;
}

__global__ void spmm_naive_elemwise_csc_gmem(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float acc = 0.0f;
	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {  // 1 LDG
		acc += get_elem_rm(a, k, y, row_idx[i]) * val[i];   // 2 LDG
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void spmm_naive_elemwise_csc_smem(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = threadIdx.x;
	uint32_t y = blockIdx.x;

	float acc = 0.0f;

	__shared__ float x_row_smem[MAT_SIZE];

	x_row_smem[x] = get_elem_rm(a, k, y, x);
	__syncthreads();

	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
		acc += x_row_smem[row_idx[i]] * val[i];
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

	__shared__ float x_row_sm[MAT_SIZE];
	__shared__ float shared_acc[MAT_SIZE];

	for (uint32_t i = x; i < k; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, y, i);
		shared_acc[i] = 0.0f;
	}
	__syncthreads();

	for (uint32_t row = 0; row < k; ++row) {
		for (uint32_t i = row_ptr[row] + x; i < row_ptr[row + 1]; i += blockDim.x) {
			atomicAdd_block(&shared_acc[col_idx[i]], x_row_sm[row] * val[i]);
		}
	}

	__syncthreads();

	for (uint32_t i = x; i < k; i += blockDim.x) {
		set_elem_rm(res, n, y, i, shared_acc[i]);
	}
}

__global__ void spmm_blocktiling_elemwise_csr(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	__shared__ float x_row_sm[MAT_SIZE];
	__shared__ float shared_acc[MAT_SIZE];

	for (size_t i = threadIdx.x; i < k; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, blockIdx.x, i);
		shared_acc[i] = 0.0f;
	}
	__syncthreads();

	for (size_t r = 0; r < k; ++r) {
		size_t bound = min(row_ptr[r + 1], row_ptr[r] + (blockIdx.y + 1) * blockDim.x);
		for (size_t i = row_ptr[r] + blockIdx.y * blockDim.x + threadIdx.x; i < bound; i += blockDim.x) {
			atomicAdd(&shared_acc[col_idx[i]], val[i] * x_row_sm[r]);
		}
	}

	__syncthreads();

	for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
		if (shared_acc[i] != 0) {
			set_elem_rm(res, n, blockIdx.x, i, shared_acc[i]);
		}
	}
}

template <const size_t N_THREADS>
__global__ void spmm_coalesced_nnzwise(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	__shared__ float x_row_sm[MAT_SIZE];

	for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, blockIdx.y, i);
	}
	__syncthreads();

	float acc = 0.0f;
	for (size_t i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += x_row_sm[row_idx[i]] * val[i];
	}
	__syncthreads();

	for (uint32_t i = WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, WARP_SIZE);
	}

	uint32_t lane_id = threadIdx.x & 0x1f;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	constexpr uint32_t n_warps = N_THREADS / WARP_SIZE;
	__shared__ float   warp_sums[n_warps];

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}

	__syncthreads();

	if (warp_id == 0) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane_id];

		constexpr uint32_t mask = 0xFF;

		for (uint32_t i = n_warps / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, WARP_SIZE);
		}

		if (lane_id == 0) {
			set_elem_rm(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

// WARN: INCOMPLETE
template <const size_t N_THREADS>
__global__ void spmm_vectorized_nnzwise_smem(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	const size_t max_nnz_per_col,
	float* __restrict__ res)
{
	constexpr size_t TK = 4;  // non-zeros assigned for each *thread*

	__shared__ float x_row_smem[MAT_SIZE];

	// for (size_t i = blockIdx.z * BK + threadIdx.x * TK; i < (blockIdx.z + 1) * BK; i += blockDim.x * TK)

	// NOTE: Coalesced acccess, plain
	// for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
	// 	x_row_sm[i] = get_elem_rm(a, k, blockIdx.y, i);
	// }

	// NOTE: Coalesced access + Vectorized loads
	for (size_t i = threadIdx.x * TK; i < MAT_SIZE; i += blockDim.x * TK) {
		const float4 f4 = reinterpret_cast<const float4*>(&a[blockIdx.y * k + i])[0];

		x_row_smem[i] = f4.x;
		x_row_smem[i + 1] = f4.y;
		x_row_smem[i + 2] = f4.z;
		x_row_smem[i + 3] = f4.w;

		// NOTE: this loop doesn't get vectorized for some reason
		// #pragma unroll
		// 		for (uint32_t t = 0; t < TK; ++t) {
		// 			// x_row_sm[i + t] = reinterpret_cast<const float*>(&tmp)[0];
		// 			// x_row_sm[i + t] = ((float*)&tmp)[0];
		// 		}
	}

	// __syncthreads();
	//
	// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
	// 	for (size_t i = 0; i < MAT_SIZE; ++i) {
	// 		printf("[%u] x_row_smem[%lu] = %.2f\n", blockIdx.z, i, x_row_smem[i]);
	// 	}
	// }
	//
	// __syncthreads();

	const size_t base_unaligned_idx = col_ptr[blockIdx.x];  // 2 LDG (but hardware performs a single load)
	const size_t col_end_idx = col_ptr[blockIdx.x + 1];     // per block

	const size_t col_nnz = col_end_idx - base_unaligned_idx;  // count[base_unaligned_idx - col_end_idx)
	const size_t block_nnz = col_nnz / gridDim.z;
	// TODO: add overflow assert
	const uint32_t block_nnz_rem = col_nnz % gridDim.z;

	/*
      * +-------------------------------------------------ROW_IDX----------------------------------------------------------+
      * +--------------------+---------------------+----------------------+--------------------+---------------------+-----+
      * | unaligned elements | vectorized elements | blockIdx.y remainder | unaligned elements | vectorized elements | ... |
      * +--------------------+---------------------+----------------------+--------------------+---------------------+-----+
      * +----------------------------blockIdx.z = 0-----------------------+--------------blockIdx.z = 1--------------+-----+
   */
	// only blockIdx.z = 1 takes account of the rem elements in blockIdx.z = 0
	// NOTE: You can do a branch here, it doesn't diverge any threads since its across the z dimension of the grid, i.e. all warps of a block will enter the same branch
	size_t row_idx_gmem_unaligned_start_idx = base_unaligned_idx + (block_nnz + (-(blockIdx.z == 1) & block_nnz_rem)) * blockIdx.z;
	size_t row_idx_gmem_aligned_start_idx = row_idx_gmem_unaligned_start_idx;
	while (!is_aligned(&row_idx[row_idx_gmem_aligned_start_idx], 16)) {
		++row_idx_gmem_aligned_start_idx;
	}
	const size_t row_idx_gmem_unaligned_cnt = row_idx_gmem_aligned_start_idx - row_idx_gmem_unaligned_start_idx;  // exclusive of block_aligned_start, because it will be vectorized

	// PERF: Having scalar_row_idx_smem next to scalar_val_smem might be a boost?
	// Having vectorized_row_idx_smem next to vectorized_val_smem might be a boost?
	// Having rem_row_idx_smem next to rem_val_smem might be a boost?
	/*
      * +--------------------------------------------------------------------------------------------------------SMEM-----------------------------------------------------------------------------------------------------------------------------+
      * +---------------------+-------------------+-------------------------+--------------------------------------------+-------------------+-----------------+-------------------+---------------------+----------------------------------------+
      * | scalar_row_idx_smem | alignment padding | vectorized_row_idx_smem | rem_row_idx_smem (only for blockIdx.z = 0) | alignment padding | scalar_val_smem | alignment padding | vectorized_val_smem | rem_val_smem (only for blockIdx.z = 0) |
      * +---------------------+-------------------+-------------------------+--------------------------------------------+-------------------+-----------------+-------------------+---------------------+----------------------------------------+
      * +--gmem_unaligned_cnt-+---------------------------------------------+---------------block_nnz_rem----------------+
      * +---------------------------block_nnz-------------------------------+
   */
	extern __shared__ __align__(16) char dyn_smem[];
	uint32_t*                            scalar_row_idx_smem = reinterpret_cast<uint32_t*>(dyn_smem);
	uint32_t*                            vectorized_row_idx_smem = reinterpret_cast<uint32_t*>(align(dyn_smem, scalar_row_idx_smem + row_idx_gmem_unaligned_cnt, 16));
	uint32_t*                            rem_row_idx_smem = vectorized_row_idx_smem + block_nnz - row_idx_gmem_unaligned_cnt;

	// PERF: at most 3 iterations -> WD
	// TODO: make into a function
	for (size_t i = row_idx_gmem_unaligned_start_idx + threadIdx.x; i < row_idx_gmem_aligned_start_idx; ++i) {
		scalar_row_idx_smem[threadIdx.x] = row_idx[i];
	}

	const size_t row_idx_block_end_idx = row_idx_gmem_unaligned_start_idx + block_nnz + block_nnz_rem;
	const size_t thread_offset = threadIdx.x * TK;

	for (size_t i = row_idx_gmem_aligned_start_idx + thread_offset, cnt = 0; i < row_idx_block_end_idx; i += blockDim.x * TK, ++cnt) {
		// TODO: Test this vs manually unrolling times
		reinterpret_cast<uint4*>(&vectorized_row_idx_smem[thread_offset + cnt * blockDim.x * TK])[0] =
			reinterpret_cast<const uint4*>(&row_idx[i])[0];
	}
	// PERF: No WD
	if (blockIdx.z == 0) {
		for (size_t i = row_idx_gmem_aligned_start_idx + block_nnz + thread_offset, cnt = 0; i < row_idx_block_end_idx; i += blockDim.x * TK, ++cnt) {
			rem_row_idx_smem[thread_offset + cnt * blockDim.x * TK] = row_idx[i];
		}
	}

	/*
      * +-------------------------------------------------VAL--------------------------------------------------------------+
      * +--------------------+---------------------+----------------------+--------------------+---------------------+-----+
      * | unaligned elements | vectorized elements | blockIdx.y remainder | unaligned elements | vectorized elements | ... |
      * +--------------------+---------------------+----------------------+--------------------+---------------------+-----+
      * +----------------------------blockIdx.z = 0-----------------------+--------------blockIdx.z = 1--------------+-----+
   */

	size_t val_gmem_unaligned_start_idx = row_idx_gmem_unaligned_start_idx;
	size_t val_gmem_aligned_start_idx = val_gmem_unaligned_start_idx;
	while (!is_aligned(&val[val_gmem_aligned_start_idx], 16)) {
		++val_gmem_aligned_start_idx;
	}
	const size_t val_gmem_unaligned_cnt = val_gmem_aligned_start_idx - val_gmem_unaligned_start_idx;  // exclusive of val_gmem_aligned_start_idx, because it will be vectorized

	/*
      * +--------------------------------------------------------------------------------------------------------SMEM-----------------------------------------------------------------------------------------------------------------------------+
      * +---------------------+-------------------+-------------------------+--------------------------------------------+-------------------+-----------------+-------------------+---------------------+----------------------------------------+
      * | scalar_row_idx_smem | alignment padding | vectorized_row_idx_smem | rem_row_idx_smem (only for blockIdx.z = 0) | alignment padding | scalar_val_smem | alignment padding | vectorized_val_smem | rem_val_smem (only for blockIdx.z = 0) |
      * +---------------------+-------------------+-------------------------+--------------------------------------------+-------------------+-----------------+-------------------+---------------------+----------------------------------------+
      * +--gmem_unaligned_cnt-+---------------------------------------------+---------------block_nnz_rem----------------+
      * +---------------------------block_nnz-------------------------------+
   */
	// the closest(rounded up) aligned address after
	// 1. the first batch of unaligned addresses
	// 2. the vectorizable addresses
	// 3. any possible scalar remainder
	float* scalar_val_smem = reinterpret_cast<float*>(align(dyn_smem, rem_row_idx_smem + block_nnz_rem, 16));
	float* vectorized_val_smem = reinterpret_cast<float*>(align(dyn_smem, scalar_val_smem + val_gmem_unaligned_cnt, 16));
	float* rem_val_smem = vectorized_val_smem + block_nnz - val_gmem_unaligned_cnt;

	// PERF: at most gridDim.z iterations -> warp divergence
	for (size_t i = val_gmem_unaligned_start_idx + threadIdx.x; i < val_gmem_aligned_start_idx; ++i) {
		scalar_val_smem[threadIdx.x] = val[i];
	}

	const size_t val_block_end_idx = val_gmem_unaligned_start_idx + block_nnz + block_nnz_rem;

	for (size_t i = val_gmem_aligned_start_idx + thread_offset, cnt = 0; i < val_block_end_idx; i += blockDim.x * TK, ++cnt) {
		reinterpret_cast<float4*>(&vectorized_val_smem[thread_offset + cnt * blockDim.x * TK])[0] =
			reinterpret_cast<const float4*>(&val[i])[0];
	}

	// PERF: No WD
	if (blockIdx.z == 0) {
		for (size_t i = val_gmem_aligned_start_idx + block_nnz + thread_offset, cnt = 0; i < val_block_end_idx; i += blockDim.x * TK, ++cnt) {
			rem_val_smem[thread_offset + cnt * blockDim.x * TK] = val[i];
		}
	}

	__syncthreads();

	// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
	// 	printf("[%u] Total nnz for this column: %u\n", blockIdx.z, col_ptr[blockIdx.x + 1] - col_ptr[blockIdx.x]);
	// 	printf("[%u] block_nnz(%lu) + block_nnz_rem(%u) = %lu\n", blockIdx.z, block_nnz, block_nnz_rem, block_nnz + block_nnz_rem);
	// 	printf("[%u] Unaligned count row_idx: %lu, Unaligned count val: %lu\n", blockIdx.z, row_idx_gmem_unaligned_cnt, val_gmem_unaligned_cnt);
	// }

	// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
	// for (size_t i = thread_offset; i < row_idx_gmem_unaligned_cnt; i += blockDim.x * TK) {
	// 	printf("[%u] scalar_row_idx_smem[%lu] = %u\n", blockIdx.z, i, scalar_row_idx_smem[i]);
	// }
	// for (size_t i = 0; i < block_nnz - row_idx_gmem_unaligned_cnt; ++i) {
	// 	printf("[%u] vectorized_row_idx_smem[%lu] = %u\n", blockIdx.z, i, vectorized_row_idx_smem[i]);
	// }
	// if (blockIdx.z == 0) {
	// 	for (size_t i = thread_offset; i < block_nnz_rem; i += blockDim.x * TK) {
	// 		printf("[%u] rem_row_idx_smem[%lu] = %u\n", blockIdx.z, i, rem_row_idx_smem[i]);
	// 	}
	// }
	// }

	// if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
	// 	// for (size_t i = thread_offset; i < val_gmem_unaligned_cnt; i += blockDim.x * TK) {
	// 	// 	printf("[%u] scalar_val_smem[%lu] = %.2f\n", blockIdx.z, i, scalar_val_smem[i]);
	// 	// }
	// 	// for (size_t i = 0; i < block_nnz - val_gmem_unaligned_cnt; ++i) {
	// 	// 	printf("[%u] vectorized_val_smem[%lu] = %.2f\n", blockIdx.z, i, vectorized_val_smem[i]);
	// 	// }
	// 	if (blockIdx.z == 0) {
	// 		for (size_t i = thread_offset; i < block_nnz_rem; i += blockDim.x * TK) {
	// 			printf("[%u] rem_row_idx_smem[%lu] = %u\n", blockIdx.z, i, rem_row_idx_smem[i]);
	// 		}
	// 	}
	// }

	__syncthreads();

	float acc = 0.0f;

	for (size_t i = thread_offset; i < row_idx_gmem_unaligned_cnt; i += blockDim.x * TK) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			printf("multiplying x_row_smem[scalar_row_idx_smem[%lu](%u)](%.2f) * scalar_val_smem[%lu](%.2f) = %.2f\n", i, scalar_row_idx_smem[i], x_row_smem[scalar_row_idx_smem[i]], i, scalar_val_smem[i], x_row_smem[scalar_row_idx_smem[i]] * scalar_val_smem[i]);
		}
		acc += x_row_smem[scalar_row_idx_smem[i]] * scalar_val_smem[i];
	}
	for (size_t i = thread_offset; i < block_nnz - row_idx_gmem_unaligned_cnt; i += blockDim.x * TK) {
		if (blockIdx.x == 0 && blockIdx.y == 0) {
			printf("multiplying x_row_smem[vectorized_row_idx_smem[%lu](%u)](%.2f) * vectorized_val_smem[%lu](%.2f) = %.2f\n", i, vectorized_row_idx_smem[i], x_row_smem[vectorized_row_idx_smem[i]], i, vectorized_val_smem[i], x_row_smem[vectorized_row_idx_smem[i]] * vectorized_val_smem[i]);
		}

		acc += x_row_smem[vectorized_row_idx_smem[i]] * vectorized_val_smem[i];
	}
	if (blockIdx.z == 0) {
		for (size_t i = thread_offset; i < block_nnz_rem; i += blockDim.x * TK) {
			if (blockIdx.x == 0 && blockIdx.y == 0) {
				printf("multiplying x_row_smem[rem_row_idx_smem[%lu](%u)](%.2f) * rem_val_smem[%lu](%.2f) = %.2f\n", i, rem_row_idx_smem[i], x_row_smem[rem_row_idx_smem[i]], i, rem_val_smem[i], x_row_smem[rem_row_idx_smem[i]] * rem_val_smem[i]);
			}
			acc += x_row_smem[rem_row_idx_smem[i]] * rem_val_smem[i];
		}
	}

	__syncwarp();

	// if (blockIdx.x == 0 && blockIdx.y == 0) {
	// 	printf("acc = %.2f\n", acc);
	// }

	for (uint32_t i = WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, WARP_SIZE);
	}

	uint32_t lane_id = threadIdx.x & 0x1f;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	constexpr uint32_t n_warps = N_THREADS / WARP_SIZE;
	__shared__ float   warp_sums[n_warps];

	// at this point the first thread (lane_id == 0) of every warp in this block
	// has the result from TK non-zeros for this col
	// this is essentially warp-wide reduction
	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// we write the warp-wide results to a block-wide memory location (SMEM)
	// so that we can perform block-wide reduction

	__syncthreads();

	// we assign the block-wide reduction to warp-0
	if (warp_id == 0) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane_id];

		constexpr uint32_t mask = 0x3;

		for (uint32_t i = n_warps / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, n_warps);
		}
		if (lane_id == 0) {
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

template <const size_t N_THREADS>
__global__ void spmm_vectorized_nnzwise_regs(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	constexpr size_t TK = 4;  // non-zeros assigned for each *thread*

	__shared__ float x_row_smem[MAT_SIZE];

	// NOTE: Coalesced acccess, plain
	// for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
	// 	x_row_sm[i] = get_elem_rm(a, k, blockIdx.y, i);
	// }

	// NOTE: Coalesced access + Vectorized loads
	for (size_t i = threadIdx.x * TK; i < MAT_SIZE; i += blockDim.x * TK) {
		const float4 f4 = reinterpret_cast<const float4*>(&a[blockIdx.y * k + i])[0];

		x_row_smem[i] = f4.x;
		x_row_smem[i + 1] = f4.y;
		x_row_smem[i + 2] = f4.z;
		x_row_smem[i + 3] = f4.w;

		// NOTE: this loop doesn't get vectorized for some reason
		// #pragma unroll
		// 		for (uint32_t t = 0; t < TK; ++t) {
		// 			// x_row_sm[i + t] = reinterpret_cast<const float*>(&tmp)[0];
		// 			// x_row_sm[i + t] = ((float*)&tmp)[0];
		// 		}
	}

	__syncthreads();

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

	const size_t           warp = threadIdx.x / WARP_SIZE;
	const size_t           lane = threadIdx.x & (WARP_SIZE - 1);
	float                  acc = 0.0f;
	__align__(16) uint32_t t_row_idx[TK] = { 0 };
	__align__(16) float    t_val[TK] = { 0.0f };

	if (blockIdx.z == 0 && warp == 0 && lane == 0) {
		for (size_t i = 0; i < ri_unaligned_cnt; ++i) {  // up to 3 iterations
			acc += x_row_smem[row_idx[base_unaligned_i + i]] * val[base_unaligned_i + i];
		}
		for (size_t i = 0; i < n_tail_loads; ++i) {  // up to 3 iterations
			acc += x_row_smem[row_idx[ri_aligned_i + gridDim.z * nnz_block + i]] * val[ri_aligned_i + gridDim.z * nnz_block + i];
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

		acc += x_row_smem[t_row_idx[0]] * t_val[0];
		acc += x_row_smem[t_row_idx[1]] * t_val[1];
		acc += x_row_smem[t_row_idx[2]] * t_val[2];
		acc += x_row_smem[t_row_idx[3]] * t_val[3];
	}

	__syncwarp();

	// if (blockIdx.x == 0 && blockIdx.y == 0) {
	// 	printf("acc = %.2f\n", acc);
	// }

	for (uint32_t i = WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, WARP_SIZE);
	}

	constexpr uint32_t n_warps = N_THREADS / WARP_SIZE;
	__shared__ float   warp_sums[n_warps];

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
	if (warp == 0) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane];

		constexpr uint32_t mask = 0x3;

		for (uint32_t i = n_warps / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, n_warps);
		}
		if (lane == 0) {
			// if (blockIdx.x == 0 && blockIdx.y == 0) {
			// 	printf("FINAL ACC: %.4f\n", acc);
			// 	printf("Before it was: %.4f\n", res[blockIdx.y * n + blockIdx.x]);
			// }
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

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

void prepare_cusparse_csr(SPMM<CSR>& spmm, CuSparse& cusparse)
{
	CUSPARSE_CHECK(cusparseCreateCsr(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

void prepare_cusparse_csc(SPMM<CSC>& spmm, CuSparse& cusparse)
{
	CUSPARSE_CHECK(cusparseCreateCsc(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

void prepare_spmm_csr(SPMM<CSR>& spmm)
{
	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
	}

	std::ifstream file_stream = { spmm.sparse_path };
	DLMCHeader    header = parse_dlmc_header(file_stream);

	size_t row_ptr_b_size = sizeof(uint32_t) * (header.n_rows + 1);
	size_t col_idx_b_size = sizeof(uint32_t) * header.nnz;
	size_t val_b_size = sizeof(float) * header.nnz;
	size_t sparse_b_size_aligned = row_ptr_b_size + calc_padding_bytes(row_ptr_b_size, ALIGNMENT_BYTES) +
	                               col_idx_b_size + calc_padding_bytes(col_idx_b_size, ALIGNMENT_BYTES) +
	                               val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);

	/**
    * Twice the total size of the dense matrices.
    * Once for the input
    * Twice for the result
    **/
	spmm.b_size = sparse_b_size_aligned + 2 * BENCHMARKING_TOTAL_DENSE_B_SIZE;
	spmm.host.data = cuda_malloc_host(spmm.b_size);
	spmm.host.d[0] = reinterpret_cast<float*>(spmm.host.data);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCHMARKING_DENSE_N_ROWS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
		}
	}

	// assert((reinterpret_cast<uintptr_t>(spmm.host.d[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.d[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.d[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.d[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.d[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	void* start_of_sparse = spmm.host.d[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] +                          // from the last ptr of spmm.host.d
	                        BENCHMARKING_DENSE_N_ROWS[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] * MAT_SIZE;  // skip 512 * 512 floats

	// start_of_sparse is 128-byte aligned guaranteed
	spmm.host.s = parse_csr_dlmc(start_of_sparse, spmm.sparse_path);

	// assert((reinterpret_cast<uintptr_t>(spmm.host.s.row_ptr) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.s.col_idx) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.s.val) & (ALIGNMENT_BYTES - 1)) == 0);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
	// assert((reinterpret_cast<uintptr_t>(spmm.host.r[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.r[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.r[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.r[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	// assert((reinterpret_cast<uintptr_t>(spmm.host.r[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	// WARN: asserts cost
	assert(sparse_b_size_aligned == spmm.host.s.b_size);

	/*
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
      * | x_32 | x_64 | x_128 | x_256 | x_512 | col_ptr | row_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
      * +------------------------------------------HOST/DEVICE------------------------------------------------+
   */

	spmm.dev.data = cuda_malloc_device(spmm.b_size);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCHMARKING_TOTAL_DENSE_B_SIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.d[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSR(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr += spmm.host.s.b_size;

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
}

void prepare_spmm_csc(SPMM<CSC>& spmm)
{
	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
	}

	std::ifstream file_stream = { spmm.sparse_path };
	DLMCHeader    header = parse_dlmc_header(file_stream);

	size_t col_ptr_b_size = sizeof(uint32_t) * (header.n_cols + 1);
	size_t row_idx_b_size = sizeof(uint32_t) * header.nnz;
	size_t val_b_size = sizeof(float) * header.nnz;
	size_t sparse_b_size_aligned = col_ptr_b_size + calc_padding_bytes(col_ptr_b_size, ALIGNMENT_BYTES) +
	                               row_idx_b_size + calc_padding_bytes(row_idx_b_size, ALIGNMENT_BYTES) +
	                               val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);

	/**
    * Twice the total size of the dense matrices.
    * Once for the input
    * Twice for the result
    **/
	spmm.b_size = sparse_b_size_aligned + 2 * BENCHMARKING_TOTAL_DENSE_B_SIZE;
	spmm.host.data = cuda_malloc_host(spmm.b_size);
	spmm.host.d[0] = reinterpret_cast<float*>(spmm.host.data);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCHMARKING_DENSE_N_ROWS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
		}
	}

	assert((reinterpret_cast<uintptr_t>(spmm.host.d[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	void* start_of_sparse = spmm.host.d[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] +                          // from the last ptr of spmm.host.d
	                        BENCHMARKING_DENSE_N_ROWS[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] * MAT_SIZE;  // skip 512 * 512 floats

	// start_of_sparse is 128-byte aligned guaranteed
	spmm.host.s = parse_csc_dlmc(start_of_sparse, spmm.sparse_path);
	spmm.host.s.max_nnz_per_col = calc_max_nnz_per_col(spmm.host.s);

	assert((reinterpret_cast<uintptr_t>(spmm.host.s.col_ptr) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.row_idx) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.val) & (ALIGNMENT_BYTES - 1)) == 0);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	// WARN: asserts cost
	assert(sparse_b_size_aligned == spmm.host.s.b_size);

	/*
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
      * | x_32 | x_64 | x_128 | x_256 | x_512 | col_ptr | row_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
      * +------------------------------------------HOST/DEVICE------------------------------------------------+
   */

	spmm.dev.data = cuda_malloc_device(spmm.b_size);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCHMARKING_TOTAL_DENSE_B_SIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.d[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSC(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr += spmm.host.s.b_size;

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
}

bool warmup_spmm_csr(SPMM<CSR>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSR>&, const uint32_t))
{
	// PERF: Bounds check
	assert(size_idx < std::size(BENCHMARKING_DENSE_N_ROWS) - 1);
	run_kernel(spmm, size_idx);

	const size_t res_size = BENCHMARKING_DENSE_N_ROWS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csr(spmm, cusparse);

	CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, cusparse.work_buffer));
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
}

bool warmup_spmm_csc(SPMM<CSC>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSC>&, const uint32_t))
{
	const size_t res_size = BENCHMARKING_DENSE_N_ROWS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemset(spmm.dev.r[size_idx], 0.0f, res_size * sizeof(float)));
	// PERF: Bounds check
	assert(size_idx < std::size(BENCHMARKING_DENSE_N_ROWS) - 1);  // DON'T REMOVE, YOU ARE DOING size_idx + 1 later
	run_kernel(spmm, size_idx);

	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csc(spmm, cusparse);

	CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, cusparse.work_buffer));
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (uint32_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	return verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
}

void run_spmm_naive_elemwise_csc_gmem(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t BN = 16;
	constexpr size_t BK = BN;

	dim3 grid(CEIL_DIV(MAT_SIZE, BN), CEIL_DIV(BENCHMARKING_DENSE_N_ROWS[idx], BK));
	dim3 block(BN, BK);

	spmm_naive_elemwise_csc_gmem<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_naive_elemwise_csc_smem(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	dim3 grid(MAT_SIZE);
	dim3 block(MAT_SIZE);

	spmm_naive_elemwise_csc_smem<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_coalesced_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	dim3 grid(MAT_SIZE);
	dim3 block(MAT_SIZE);

	spmm_coalesced_elemwise_csr<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_blocktiling_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t BN = 32;
	constexpr size_t TN = 4;

	dim3 grid(m, n / BN);
	dim3 block(CEIL_DIV(BN, TN));

	spmm_blocktiling_elemwise_csr<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_coalesced_nnzwise(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 64;

	dim3 grid(n, BENCHMARKING_DENSE_N_ROWS[idx]);
	dim3 block(n_threads);

	spmm_coalesced_nnzwise<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

// WARN: INCOMPLETE
// void run_spmm_vectorized_nnzwise_smem(SPMM<CSC>& spmm, const uint32_t idx)
// {
// 	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
// 	const size_t k = spmm.dev.s.rows;
// 	const size_t n = spmm.dev.s.cols;
//
//   constexpr size_t n_threads = 64;
//
// 	// // PERF: Hack ~ find a better way to deal with having to add instead of set the result
// 	// const size_t res_size = BENCHMARKING_DENSE_N_ROWS[idx] * MAT_SIZE * sizeof(float);
// 	// // BUG: Hardcoded value
// 	// // ensures there is enough space for alignment
// 	// // doing 4 alignments at 16-byte each
// 	// const size_t padding = 4 * 16;
// 	// const size_t dyn_mem_b_size = spmm.host.s.max_nnz_per_col * sizeof(float) + spmm.host.s.max_nnz_per_col * sizeof(uint32_t) + padding;  // + TK for ensuring alignment of val_smem
// 	// CUDA_CHECK(cudaMemset(spmm.dev.r[idx], 0, res_size));
// 	// CUDA_CHECK(cudaDeviceSynchronize());
// 	// dim3 grid(n, BENCHMARKING_DENSE_N_ROWS[idx], k / 256);
// 	// dim3 block(64);
// 	// spmm_csc_2d_blocktiling<<<grid, block, dyn_mem_b_size>>>(
// 	// 	spmm.dev.d[idx],
// 	// 	spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
// 	// 	m, k, n, spmm.host.s.max_nnz_per_col, spmm.dev.r[idx]);
//
//   dim3 grid();
//   dim3 block();
//
//   spmm_vectorized_nnzwise_smem<n_threads>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, , float *__restrict res)
// }

void run_spmm_vectorized_nnzwise_regs(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 64;
	constexpr size_t BK = 512;

	dim3 grid(n, m, CEIL_DIV(MAT_SIZE, BK));
	dim3 block(n_threads);

	spmm_vectorized_nnzwise_regs<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

// void prepare_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	// mhsa_load_host_csc(mhsa, mhsa.config, mhsa.dlmc, mhsa.weights);
//
// 	// TODO: Find a better name
// 	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
// 	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;
//
// 	size_t res_b_size = sizeof(float) * (kv_size * 4 + gemm_res_size * 2 + 1);  // Q, K, V, gemm result, float acc for softmax, Attention matrix, Final Result
//
// 	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
// 	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));
//
// 	/*
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * | x | w_q | w_k | w_v | w_o | mask | Q | K | V | QK^T | ACC | A | Final Result |
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * +-------------HOST-----------------+----------------DEVICE---------------------+
//    */
//
// 	res.x = reinterpret_cast<float*>(mhsa.dev);
// 	size_t b_x_size = sizeof(float) * kv_size;
//
// 	char* ptr = reinterpret_cast<char*>(res.x) + b_x_size;
//
// 	// TODO: This call copy assignment operator of CSC
// 	// check if the custom one does what you want
// 	res.w_q = mhsa.weights.w_q[0];
// 	res.w_q.partition(ptr);
// 	ptr += res.w_q.b_size;
//
// 	res.w_k = mhsa.weights.w_k[0];
// 	res.w_k.partition(ptr);
// 	ptr += res.w_k.b_size;
//
// 	res.w_v = mhsa.weights.w_v[0];
// 	res.w_v.partition(ptr);
// 	ptr += res.w_v.b_size;
//
// 	res.w_o = mhsa.weights.w_o[0];
// 	res.w_o.partition(ptr);
// 	ptr += res.w_o.b_size;
//
// 	res.q_res = reinterpret_cast<float*>(ptr);
// 	res.k_res = res.q_res + kv_size;
// 	res.v_res = res.k_res + kv_size;
// 	res.gemm_res = res.v_res + kv_size;
// 	res.softmax_acc = res.gemm_res + gemm_res_size;
// 	res.attention = res.softmax_acc + 1;
//
// 	return res;
// }
//
// void run_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	DevMHSA      d = prepare_mhsa(mhsa);
// 	const size_t m = mhsa.config.input_sequence_size;
// 	const size_t n = d.w_q.cols;
//
// 	// One thread per element of the output
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_gm(32, 32);
// 	dim3 spmm_grid_gm(
// 		(n + spmm_block_gm.x - 1) / spmm_block_gm.x,
// 		(m + spmm_block_gm.y - 1) / spmm_block_gm.y);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_sm(512);
// 	dim3 spmm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x32)=(32x32)
// 	dim3 gemm_block_sm(32);
// 	dim3 gemm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x32)
// 	dim3 softmax_block(32, 32);
// 	dim3 softmax_grid(
// 		(m + softmax_block.x - 1) / softmax_block.x,
// 		(m + softmax_block.y - 1) / softmax_block.y);  // This should actually be equal to (1,1) i.e. one block
//
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_q.col_ptr, d.w_q.row_idx, d.w_q.val, mhsa.config.input_sequence_size, d.w_q.rows, d.w_q.cols, d.q_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_k.col_ptr, d.w_k.row_idx, d.w_k.val, mhsa.config.input_sequence_size, d.w_k.rows, d.w_k.cols, d.k_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_v.col_ptr, d.w_v.row_idx, d.w_v.val, mhsa.config.input_sequence_size, d.w_v.rows, d.w_v.cols, d.v_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	gemm<<<gemm_grid_sm, gemm_block_sm>>>(d.q_res, d.k_res, mhsa.config.input_sequence_size, d.w_q.rows, mhsa.config.input_sequence_size, d.gemm_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	softmax<<<softmax_grid, softmax_block>>>(d.gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, d.softmax_acc, d.attention);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// TODO: can this be async?
// 	// TODO: THIS NEEDS TO WRITE TO PAGE-LOCKED MEMORY NOT SOME RANDOM ALLOCATED MEMORY
// 	//
// 	// CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
// }
