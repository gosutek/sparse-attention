#include "allocator.h"
#include "cuda_allocator.cuh"
#include "helpers.h"
#include "matrix.h"
#include "spmm.h"

// TODO: Maybe copy to a contiguous mem block in host first then in dev
// for further optimization

static SpmmInternalStatus_t _d_sp_copy(DevArena* const arena, SpMatDescr* sp)
{
	uint8_t* d_ptr = arena->d_ptr;

	const uint64_t sp_bsize = sp_mat_byte_size_get(sp);
	if (mem_arena_dev_push(arena, sp_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	const uint64_t ptr_bsize = sp_mat_ptr_count_get(sp);
	const uint64_t idx_bsize = sp_mat_idx_bytes_get(sp);
	const uint64_t val_bsize = sp_mat_val_bytes_get(sp);

	switch (sp->format) {
	case SPARSE_FORMAT_CSR:
		cudaMemcpy(d_ptr, sp->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, sp->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		d_ptr += idx_bsize;

		cudaMemcpy(d_ptr, sp->val, val_bsize, cudaMemcpyHostToDevice);
		break;
	case SPARSE_FORMAT_CSC:
		cudaMemcpy(d_ptr, sp->csr.row_ptr, ptr_bsize, cudaMemcpyHostToDevice);
		d_ptr += ptr_bsize;

		cudaMemcpy(d_ptr, sp->csr.col_idx, idx_bsize, cudaMemcpyHostToDevice);
		d_ptr += idx_bsize;

		cudaMemcpy(d_ptr, sp->val, val_bsize, cudaMemcpyHostToDevice);
		break;
	}
}

static SpmmInternalStatus_t _d_dn_copy(DevArena* const arena, DnMatDescr* dn)
{
	uint8_t* d_ptr = arena->d_ptr;

	const uint64_t dn_bsize = dn_mat_bytes_get(dn);
	if (mem_arena_dev_push(arena, dn_bsize, reinterpret_cast<void**>(&d_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}
	cudaMemcpy(d_ptr, dn->val, dn_bsize, cudaMemcpyHostToDevice);

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmStatus_t spmm(ExecCtx* ctx, SpMatDescr_t sp, DnMatDescr_t dn)
{
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (!ctx->dev_arena.d_ptr) {
		mem_arena_dev_create(&ctx->dev_arena, GIB(1));
	}

	// TODO: Error check these two
	_d_sp_copy(&ctx->dev_arena, sp);
	_d_dn_copy(&ctx->dev_arena, dn);

	const uint64_t res_bsize = spmm_res_mat_bytes_get(sp, dn);
	uint8_t*       res_ptr = nullptr;

	if (mem_arena_dev_push(&ctx->dev_arena, res_bsize, reinterpret_cast<void**>(&res_ptr)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}

	// SPMM
}
