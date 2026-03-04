#include "allocator.h"
#include "cuda/kernels/spmm_csr.cuh"
#include "cuda_allocator.cuh"
#include "helpers.h"
#include "matrix.h"
#include "spmm.h"

// TODO: Maybe copy to a contiguous mem block in host first then in dev
// for further optimization
static SpmmInternalStatus_t _d_sp_copy(DevArena* const arena, SpMatDescr* const dst, const SpMatDescr* const src)
{
	uint8_t* d_ptr = arena->d_ptr;

	const uint64_t sp_bsize = sp_mat_byte_size_get(src);
	if (mem_arena_dev_push(arena, sp_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	const uint64_t ptr_bsize = sp_mat_ptr_count_get(src);
	const uint64_t idx_bsize = sp_mat_idx_bytes_get(src);
	const uint64_t val_bsize = sp_mat_val_bytes_get(src);

	dst->format = src->format;
	dst->rows = src->rows;
	dst->cols = src->cols;
	dst->nnz = src->nnz;

	switch (src->format) {
	case SPARSE_FORMAT_CSR:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csr.row_ptr = reinterpret_cast<uint32_t*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csr.col_idx = reinterpret_cast<uint32_t*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	case SPARSE_FORMAT_CSC:
		cudaMemcpy(d_ptr, src->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		dst->csc.col_ptr = reinterpret_cast<uint32_t*>(d_ptr);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, src->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		dst->csc.row_idx = reinterpret_cast<uint32_t*>(d_ptr);
		d_ptr += idx_bsize;

		break;
	}
	CUDA_CHECK(cudaMemcpy(d_ptr, src->val, val_bsize, cudaMemcpyHostToDevice));
	dst->val = reinterpret_cast<float*>(d_ptr);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

static SpmmInternalStatus_t _d_dn_copy(DevArena* const arena, DnMatDescr* dst, DnMatDescr* src)
{
	uint8_t* d_ptr = arena->d_ptr;

	const uint64_t dn_bsize = dn_mat_bytes_get(src);
	if (mem_arena_dev_push(arena, dn_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	dst->format = src->format;
	dst->rows = src->rows;
	dst->cols = src->cols;

	CUDA_CHECK(cudaMemcpy(d_ptr, src->val, dn_bsize, cudaMemcpyHostToDevice));
	dst->val = reinterpret_cast<float*>(d_ptr);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmStatus_t spmm(ExecCtx* ctx, SpMatDescr_t h_sp, DnMatDescr_t h_dn)
{
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (!ctx->dev_arena.d_ptr && mem_arena_dev_create(&ctx->dev_arena, GIB(1)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}

	// TODO: Error check these two
	SpMatDescr d_sp;
	_d_sp_copy(&ctx->dev_arena, &d_sp, h_sp);

	DnMatDescr d_dn;
	_d_dn_copy(&ctx->dev_arena, &d_dn, h_dn);

	const uint64_t res_bsize = spmm_res_mat_bytes_get(h_sp, h_dn);
	float*         res_ptr = nullptr;

	if (mem_arena_dev_push(&ctx->dev_arena, res_bsize, reinterpret_cast<void**>(&res_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}

	constexpr size_t BM = 8;
	constexpr size_t BK = BM;

	const uint32_t res_rows = h_sp->rows;
	const uint32_t res_cols = h_dn->cols;

	static_assert(BM <= 32);  // otherwise threads per block exceed max
	dim3 grid(CEIL_DIVI(res_cols, BK), CEIL_DIVI(res_rows, BM));
	dim3 block(BK, BM);

	// _k_spmm_naive_elemwise_gmem_csr<<<grid, block>>>(d_sp.csr.row_ptr, d_sp.csr.col_idx, d_sp.val, d_dn.val, d_sp.rows, d_sp.cols, d_dn.cols, res_ptr);

	return SPMM_STATUS_SUCCESS;
}
