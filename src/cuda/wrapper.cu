#include "allocator.h"
#include "cuda/helpers.cuh"
#include "cuda/kernels/spmm.cuh"
#include "cuda_allocator.cuh"
#include "helpers.h"
#include "matrix.h"
#include "spmm.h"

// TODO: Maybe copy to a contiguous mem block in host first then in dev
// for further optimization
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
	case SPARSE_FORMAT_CSR:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csr.row_ptr = reinterpret_cast<u32*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csr.col_idx = reinterpret_cast<u32*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	case SPARSE_FORMAT_CSC:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csc.col_ptr = reinterpret_cast<u32*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csc.row_idx = reinterpret_cast<u32*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	}
	CUDA_CHECK(cudaMemcpy(d_ptr, src->val, val_bsize, cudaMemcpyHostToDevice));
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

	CUDA_CHECK(cudaMemcpy(d_ptr, src->val, dn_bsize, cudaMemcpyHostToDevice));
	dst->val = reinterpret_cast<f32*>(d_ptr);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmStatus_t spmm(ExecCtx* ctx, SpMatDescr_t h_sp, DnMatDescr_t h_dn, DnMatDescr_t h_res, SpmmKernelType_t kernel_type, SpmmInvert_t invert)
{
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (!ctx->dev_arena._d_ptr && mem_arena_dev_create(&ctx->dev_arena, GIB(1)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}

	// TODO: Error check these two
	SpMatDescr d_sp;
	_d_sp_copy(&ctx->dev_arena, &d_sp, h_sp);

	DnMatDescr d_dn;
	_d_dn_copy(&ctx->dev_arena, &d_dn, h_dn);

	DnMatDescr d_res = {
		.format = DENSE_FORMAT_ROW_MAJOR,
		.rows = d_sp.rows,
		.cols = d_dn.cols,
		.val = nullptr
	};
	const u64 res_bsize = dn_mat_bytes_get(&d_res);

	if (mem_arena_dev_push(&ctx->dev_arena, res_bsize, reinterpret_cast<void**>(&d_res.val)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}
	// d_res.val now points to device memory

	switch (kernel_type) {
	case SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK:
		{
			constexpr u32 BM = 8;
			constexpr u32 BK = BM;

			const u32 res_rows = h_sp->rows;
			const u32 res_cols = h_dn->cols;

			static_assert(BM <= 32);  // otherwise threads per block exceed max
			dim3 grid(CEIL_DIVI(res_cols, BK), CEIL_DIVI(res_rows, BM));
			dim3 block(BK, BM);

			if (invert == SPMM_KERNEL_NO_INVERT) {
				_k_spmm_naive_elemwise_gmem<<<grid, block>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_dn.cols, d_res.val);
			} else {
				_k_ispmm_naive_elemwise_gmem<<<grid, block>>>(d_dn.val, d_sp.csc.col_ptr, d_sp.csc.row_idx, d_sp.val, d_dn.rows, d_dn.cols, d_sp.cols, d_res.val);
			}
			CUDA_CHECK(cudaDeviceSynchronize());

			break;
		}
	case SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM:
		{
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3     grid(d_dn.cols);
				const dim3     block(d_sp.rows);
				const u64 smem_bsize = d_sp.rows * sizeof *d_sp.val;
				_k_spmm_naive_elemwise_smem<<<grid, block, smem_bsize>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_sp.cols, d_res.val);
			} else {
				const dim3     grid(d_dn.rows);
				const dim3     block(d_sp.cols);
				const u64 smem_bsize = d_dn.cols * sizeof *d_dn.val;
				_k_ispmm_naive_elemwise_smem<<<grid, block, smem_bsize>>>(d_dn.val, d_sp.csc.col_ptr, d_sp.csc.row_idx, d_sp.val, d_dn.rows, d_dn.cols, d_sp.cols, d_res.val);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_COALESCED:
		{
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(d_dn.cols, d_sp.rows);
				const dim3 block(64);

				const u64 smem_bsize = (d_dn.rows + block.x / _CONSTANTS_WARP_SIZE) * sizeof *d_dn.val;

				_k_spmm_coalesced_nnzwise<<<grid, block, smem_bsize>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_dn.cols, d_res.val);
			} else {
				const dim3 grid(d_sp.cols, d_dn.rows);
				const dim3 block(64);

				const u64 smem_bsize = (d_dn.cols + block.x /*thread_cnt*/ / _CONSTANTS_WARP_SIZE) * sizeof *d_dn.val;

				_k_ispmm_coalesced_nnzwise<<<grid, block, smem_bsize>>>(d_dn.val, d_sp.csc.col_ptr, d_sp.csc.row_idx, d_sp.val, d_dn.rows, d_dn.cols, d_sp.cols, d_res.val);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_COALESCED_NO_SMEM:
		{
			const dim3     block(32);
			const u64 smem_bsize = (block.x / _CONSTANTS_WARP_SIZE) * sizeof *d_dn.val;
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(d_dn.cols, d_sp.rows);

				_k_spmm_coalesced_nnzwise_no_smem<<<grid, block, smem_bsize>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_dn.cols, d_res.val);
			} else {
				const dim3 grid(d_sp.cols, d_dn.rows);

				_k_ispmm_coalesced_nnzwise_no_smem<<<grid, block, smem_bsize>>>(d_dn.val, d_sp.csc.col_ptr, d_sp.csc.row_idx, d_sp.val, d_dn.rows, d_dn.cols, d_sp.cols, d_res.val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_VECTORIZED:
		{
			constexpr u32 BK = 512;
			const dim3         block(64);
			const u64     smem_bsize = (block.x / _CONSTANTS_WARP_SIZE) * sizeof *d_dn.val;
			if (invert == SPMM_KERNEL_NO_INVERT) {
				const dim3 grid(d_dn.cols, d_sp.rows, CEIL_DIVI(d_sp.cols, BK));

				_k_spmm_vectorized_nnzwise_regs<<<grid, block, smem_bsize>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_dn.cols, d_res.val);
			} else {
				const dim3 grid(d_sp.cols, d_dn.rows, CEIL_DIVI(d_dn.cols, BK));

				_k_ispmm_vectorized_nnzwise_regs<<<grid, block, smem_bsize>>>(d_dn.val, d_sp.csc.col_ptr, d_sp.csc.row_idx, d_sp.val, d_dn.rows, d_dn.cols, d_sp.cols, d_res.val);
			}
			break;
		}
	case SPMM_KERNEL_TYPE_NNZWISE_FINAL:
		return SPMM_STATUS_INTERNAL_ERROR;
		break;
	}

	CUDA_CHECK(cudaMemcpy(h_res->val, d_res.val, res_bsize, cudaMemcpyDeviceToHost));

	mem_arena_dev_pop(&ctx->dev_arena, res_bsize);

	return SPMM_STATUS_SUCCESS;
}
