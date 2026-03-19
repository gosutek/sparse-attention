#include <cstdio>

#include "allocator.h"
#include "cuda_helpers.cuh"
#include "kernels/spmm.cuh"
#include "matrix.h"
#include "spmm.h"

static SpmmInternalStatus_t _d_sp_copy(DevArena* const arena, SpMatDescr* const dst, const SpMatDescr* const src)
{
	uint8_t* d_ptr = NULL;

	const u64 sp_bsize = sp_mat_byte_size_get(src);
	if (mem_arena_dev_push(arena, sp_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	const u64 ptr_bsize = sp_mat_ptr_bytes_get(src);
	const u64 idx_bsize = sp_mat_idx_bytes_get(src);
	const u64 val_bsize = sp_mat_val_bytes_get(src);

	dst->format = src->format;
	dst->rows = src->rows;
	dst->cols = src->cols;
	dst->nnz = src->nnz;

	switch (src->format) {
	case SPMM_FORMAT_SPARSE_CSR:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csr.row_ptr = reinterpret_cast<u32*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csr.col_idx = reinterpret_cast<u32*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	case SPMM_FORMAT_SPARSE_CSC:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csc.col_ptr = reinterpret_cast<u32*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csc.row_idx = reinterpret_cast<u32*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	}
	CHECK_CUDA(cudaMemcpy(d_ptr, src->val, val_bsize, cudaMemcpyHostToDevice));
	dst->val = reinterpret_cast<f32*>(d_ptr);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

static SpmmInternalStatus_t _d_dn_copy(DevArena* const arena, DnMatDescr* dst, DnMatDescr* src)
{
	uint8_t* d_ptr = NULL;

	const u64 dn_bsize = dn_mat_bytes_get(src);
	if (mem_arena_dev_push(arena, dn_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	dst->format = src->format;
	dst->rows = src->rows;
	dst->cols = src->cols;

	CHECK_CUDA(cudaMemcpy(d_ptr, src->val, dn_bsize, cudaMemcpyHostToDevice));
	dst->val = reinterpret_cast<f32*>(d_ptr);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmStatus_t spmm(ExecCtx* ctx, SpMatDescr_t sp, DnMatDescr_t dn, DnMatDescr_t res, SpmmKernelType_t kernel_type, SpmmInvert_t invert)
{
	// TODO: Handle ctx->dev_arena being null, ctx->dev_arena._d_ptr being null && not being a valid device pointer
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	switch (kernel_type) {
	case SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK:
		{
			constexpr u32 BM = 8;
			constexpr u32 BK = BM;

			const u32 res_rows = sp->rows;
			const u32 res_cols = dn->cols;

			static_assert(BM <= 32);  // otherwise threads per block exceed max
			dim3 grid(CEIL_DIVI(res_cols, BK), CEIL_DIVI(res_rows, BM));
			dim3 block(BK, BM);

			if (invert == SPMM_KERNEL_NO_INVERT) {
				_k_spmm_naive_elemwise_gmem<<<grid, block>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, dn->cols, res->val);
			} else {
				_k_ispmm_naive_elemwise_gmem<<<grid, block>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, res->val);
			}

			break;
		}
	case SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM:
		{
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(dn->cols);
				const dim3 block(sp->rows);
				const u64  smem_bsize = sp->rows * sizeof *sp->val;
				_k_spmm_naive_elemwise_smem<<<grid, block, smem_bsize>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, sp->cols, res->val);
			} else {
				const dim3 grid(dn->rows);
				const dim3 block(sp->cols);
				const u64  smem_bsize = dn->cols * sizeof *dn->val;
				_k_ispmm_naive_elemwise_smem<<<grid, block, smem_bsize>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, res->val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_COALESCED:
		{
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(dn->cols, sp->rows);
				const dim3 block(64);

				const u64 smem_bsize = (dn->rows + block.x / _CONSTANTS_WARP_SIZE) * sizeof *dn->val;

				_k_spmm_coalesced_nnzwise<<<grid, block, smem_bsize>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, dn->cols, res->val);
			} else {
				const dim3 grid(sp->cols, dn->rows);
				const dim3 block(64);

				const u64 smem_bsize = (dn->cols + block.x /*thread_cnt*/ / _CONSTANTS_WARP_SIZE) * sizeof *dn->val;

				_k_ispmm_coalesced_nnzwise<<<grid, block, smem_bsize>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, res->val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_COALESCED_NO_SMEM:
		{
			const dim3 block(32);
			const u64  smem_bsize = (block.x / _CONSTANTS_WARP_SIZE) * sizeof *dn->val;
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(dn->cols, sp->rows);

				_k_spmm_coalesced_nnzwise_no_smem<<<grid, block, smem_bsize>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, dn->cols, res->val);
			} else {
				const dim3 grid(sp->cols, dn->rows);

				_k_ispmm_coalesced_nnzwise_no_smem<<<grid, block, smem_bsize>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, res->val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_VECTORIZED:
		{
			constexpr u32 BK = 512;
			const dim3    block(64);
			const u64     smem_bsize = (block.x / _CONSTANTS_WARP_SIZE) * sizeof *dn->val;
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(dn->cols, sp->rows, CEIL_DIVI(sp->cols, BK));

				_k_spmm_vectorized_nnzwise_regs<<<grid, block, smem_bsize>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, dn->cols, res->val);
			} else {
				const dim3 grid(sp->cols, dn->rows, CEIL_DIVI(dn->cols, BK));

				_k_ispmm_vectorized_nnzwise_regs<<<grid, block, smem_bsize>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, res->val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_COLUMN_TILING_NNZWISE:
		{
			const dim3 block(32);

			const u64 smem_bsize = (block.x / _CONSTANTS_WARP_SIZE) * sizeof *dn->val;

			if (invert == SPMM_KERNEL_NO_INVERT) {
				constexpr u32 BM = 16;
				const dim3    grid(CEIL_DIVI(sp->rows, BM), dn->cols);

				_k_spmm_column_tiling_nnzwise<<<grid, block, smem_bsize>>>(sp->csr.row_ptr, sp->csr.col_idx, sp->val, dn->val, sp->rows, sp->cols, dn->cols, BM, res->val);
			} else {
				constexpr u32 BN = 16;
				const dim3    grid(CEIL_DIVI(sp->cols, BN), dn->rows);

				_k_ispmm_column_tiling_nnzwise<<<grid, block, smem_bsize>>>(dn->val, sp->csc.col_ptr, sp->csc.row_idx, sp->val, dn->rows, dn->cols, sp->cols, BN, res->val);
			}
			break;
		}
	}

	const u64 res_bsize = dn_mat_bytes_get(res);
	mem_arena_dev_pop(&ctx->dev_arena, res_bsize);
	const u64 dn_bsize = dn_mat_bytes_get(dn);
	mem_arena_dev_pop(&ctx->dev_arena, dn_bsize);
	const u64 sp_bsize = sp_mat_byte_size_get(sp);
	mem_arena_dev_pop(&ctx->dev_arena, sp_bsize);

	return SPMM_STATUS_SUCCESS;
}
