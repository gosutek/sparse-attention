#include <string.h>

#include "allocator.h"
#include "cuda_mem_wrapper.cuh"
#include "matrix.h"
#include "spmm.h"

static SpmmStatus_t get_max_nnz_per_row(size_t* const max_nnz_out, SpMatDescr* const sp_mat_descr)
{
	if (!sp_mat_descr || !max_nnz_out) {
		return SPMM_STATUS_INVALID_VALUE;
	}
	*max_nnz_out = 0;
	const size_t row_ptr_count = sp_mat_ptr_count_get(sp_mat_descr);

	for (size_t i = 0; i < row_ptr_count - 1; ++i) {
		const size_t curr_col_nnz = ((u32*)(sp_mat_descr->csr.row_ptr))[i + 1] - ((u32*)(sp_mat_descr->csr.row_ptr))[i];
		*max_nnz_out = *max_nnz_out > curr_col_nnz ? *max_nnz_out : curr_col_nnz;
	}
	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csr_to_row_major(SpMatDescr_t sp, DnMatDescr_t dn)
{
	if (!sp || !dn) {  // I except both of these to be allocated
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (sp->format != SPMM_FORMAT_SPARSE_CSR) {
		return SPMM_STATUS_NOT_SUPPORTED;
	}

	// INFO: Will need to change when you template out types
	memset(dn->val, 0, sp->rows * sp->cols * (sizeof(f32)));

	for (size_t i = 0; i < sp->rows; ++i) {
		const u32* const row_ptr = sp->csr.row_ptr;
		for (u32 j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
			dn->val[i * sp->cols + sp->csr.col_idx[j]] = sp->val[j];
		}
	}
	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csc_to_col_major(SpMatDescr_t sp, DnMatDescr_t dn)
{
	if (!sp || !dn) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (sp->format != SPMM_FORMAT_SPARSE_CSC) {
		return SPMM_STATUS_NOT_SUPPORTED;
	}

	// INFO: Will need to change when you template out types
	memset(dn->val, 0, sp->rows * sp->cols * (sizeof(f32)));

	for (size_t i = 0; i < sp->cols; ++i) {
		const u32* const col_ptr = sp->csc.col_ptr;
		for (u32 j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
			dn->val[i * sp->rows + sp->csc.row_idx[j]] = sp->val[j];
		}
	}

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr, SpMatDescr_t sp_csc)
{
	if (!ctx || !sp_csr || !sp_csc) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	memset(sp_csc->csc.col_ptr, 0, (sp_csc->cols + 1) * (sizeof *(sp_csc->csc.col_ptr)));

	for (u32 i = 0; i < sp_csc->nnz; ++i) {
		sp_csc->csc.col_ptr[sp_csr->csr.col_idx[i] + 1]++;
	}

	for (u32 i = 1; i < sp_csc->cols + 1; ++i) {
		// Prefix sum the preprocessed csc.col_ptr
		sp_csc->csc.col_ptr[i] += sp_csc->csc.col_ptr[i - 1];
	}

	// INFO: csc.col_ptr -> ready

	void*     work_buffer = NULL;
	const u64 work_buffer_bsize = (sp_csc->cols + 1) * (sizeof *(sp_csc->csc.col_ptr));
	if (mem_arena_host_push((MemArena*)ctx, work_buffer_bsize, &work_buffer) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}
	memcpy(work_buffer, sp_csc->csc.col_ptr, work_buffer_bsize);

	for (u32 row = 0; row < sp_csr->rows; ++row) {
		for (u32 i = sp_csr->csr.row_ptr[row]; i < sp_csr->csr.row_ptr[row + 1]; ++i) {
			// TODO: Change when/if you template out the type
			const u32 col = sp_csr->csr.col_idx[i];
			const u32 dest_pos = ((u32*)(work_buffer))[col]++;
			sp_csc->csc.row_idx[dest_pos] = row;
			sp_csc->val[dest_pos] = sp_csr->val[i];
		}
	}

	mem_arena_host_pop((MemArena*)ctx, work_buffer_bsize);

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csc_to_csr(ExecutionContext_t ctx, SpMatDescr_t sp_csc, SpMatDescr_t sp_csr)
{
	if (!ctx || !sp_csr || !sp_csc) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	memset(sp_csr->csr.row_ptr, 0, (sp_csc->rows + 1) * (sizeof *(sp_csr->csr.row_ptr)));

	for (u32 i = 0; i < sp_csr->nnz; ++i) {
		sp_csr->csr.row_ptr[sp_csc->csc.row_idx[i] + 1]++;
	}

	for (u32 i = 1; i < sp_csr->nnz; ++i) {
		// Prefix sum the preprocessed csc.col_ptr
		sp_csr->csr.row_ptr[i] += sp_csr->csr.row_ptr[i - 1];
	}

	// INFO: csr.row_ptr -> ready

	void*     work_buffer = NULL;
	const u64 work_buffer_bsize = (sp_csr->rows + 1) * (sizeof *(sp_csr->csr.row_ptr));
	if (mem_arena_host_push((MemArena*)ctx, work_buffer_bsize, &work_buffer) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_INTERNAL_ERROR;
	}
	memcpy(work_buffer, sp_csr->csr.row_ptr, work_buffer_bsize);

	for (u32 col = 0; col < sp_csc->cols; ++col) {
		for (u32 i = sp_csc->csc.col_ptr[col]; i < sp_csc->csc.col_ptr[col + 1]; ++i) {
			// TODO: Change when/if you template out the type
			const u32 row = sp_csc->csc.row_idx[i];
			const u32 dest_pos = ((u32*)(work_buffer))[row]++;
			sp_csr->csr.col_idx[dest_pos] = col;
			sp_csr->val[dest_pos] = sp_csc->val[i];
		}
	}

	mem_arena_host_pop((MemArena*)ctx, work_buffer_bsize);

	return SPMM_STATUS_SUCCESS;
}

// TODO: Make a COO descriptor
/*
 * Converts mtx from COO to CSR format
 */
// void coo_to_csr(COOMatrix& mtx)
// {
// 	std::vector<int>      row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
// 	std::vector<u32> col_idx(static_cast<size_t>(mtx.nnz));
//
// 	std::vector<f32> val(static_cast<size_t>(mtx.nnz));
//
// 	std::sort(mtx.elements.begin(), mtx.elements.end(), [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });
//
// 	for (size_t i = 0; i < mtx.elements.size(); ++i) {
// 		const auto& e = mtx.elements[i];
// 		row_ptr[static_cast<size_t>(e.row) + 1]++;
// 		col_idx[i] = e.col;
// 		val[i] = e.val;
// 	}
// 	std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.data());
//
// 	return;
// }

f32 measure_sparsity(void* s, u32 size)
{
	f32* ptr = (f32*)(s);
	f32  nz = .0f;
	for (u32 i = 0; i < size; i++) {
		if (ptr[i] == 0)
			nz++;
	}
	return nz / (f32)(size);
}

SpmmStatus_t sp_csr_create(ExecutionContext_t ctx, SpMatDescr_t* sp,
	u32  rows,
	u32  cols,
	u32  nnz,
	u32* row_ptr,
	u32* col_idx,
	f32* val)
{
	if (sp == NULL || *sp != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)ctx, sizeof **sp, (void**)sp) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp)->format = SPMM_FORMAT_SPARSE_CSR;

	(*sp)->rows = rows;
	(*sp)->cols = cols;
	(*sp)->nnz = nnz;

	const u64 row_ptr_bsize = (rows + 1) * sizeof *row_ptr;
	const u64 col_idx_bsize = nnz * sizeof *col_idx;
	const u64 val_bsize = nnz * sizeof *val;
	const u64 bsize = row_ptr_bsize + col_idx_bsize + val_bsize;
	if (mem_arena_dev_push(&ctx->dev_arena, bsize, (void**)&(*sp)->csr.row_ptr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp)->csr.col_idx = (*sp)->csr.row_ptr + rows + 1;
	(*sp)->val = (float*)((*sp)->csr.col_idx + nnz);

	cuda_mem_cpy_hd((*sp)->csr.row_ptr, row_ptr, row_ptr_bsize);
	cuda_mem_cpy_hd((*sp)->csr.col_idx, col_idx, col_idx_bsize);
	cuda_mem_cpy_hd((*sp)->val, val, val_bsize);

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csr_get(SpMatDescr* sp,
	u32*                            rows,
	u32*                            cols,
	u32*                            nnz,
	u32**                           row_ptr,
	u32**                           col_idx,
	f32**                           val)
{
	if (!sp) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (sp->format != SPMM_FORMAT_SPARSE_CSR) {
		return SPMM_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
	}

	*rows = sp->rows;
	*cols = sp->cols;
	*nnz = sp->nnz;
	*row_ptr = sp->csr.row_ptr;
	*col_idx = sp->csr.col_idx;
	*val = sp->val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csc_create(ExecutionContext_t ctx, SpMatDescr_t* sp,
	u32  rows,
	u32  cols,
	u32  nnz,
	u32* col_ptr,
	u32* row_idx,
	f32* val)
{
	if (sp == NULL || *sp != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **sp, (void**)sp) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp)->format = SPMM_FORMAT_SPARSE_CSC;

	(*sp)->rows = rows;
	(*sp)->cols = cols;
	(*sp)->nnz = nnz;

	const u64 col_ptr_bsize = (cols + 1) * sizeof *col_ptr;
	const u64 row_idx_bsize = nnz * sizeof *row_idx;
	const u64 val_bsize = nnz * sizeof *val;
	const u64 bsize = col_ptr_bsize + row_idx_bsize + val_bsize;
	if (mem_arena_dev_push(&ctx->dev_arena, bsize, (void**)&(*sp)->csc.col_ptr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp)->csc.row_idx = row_idx + cols + 1;
	(*sp)->val = (float*)((*sp)->csc.row_idx + nnz);

	cuda_mem_cpy_hd((*sp)->csc.col_ptr, col_ptr, col_ptr_bsize);
	cuda_mem_cpy_hd((*sp)->csc.row_idx, row_idx, row_idx_bsize);
	cuda_mem_cpy_hd((*sp)->val, val, val_bsize);

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csc_get(SpMatDescr* sp,
	u32*                            rows,
	u32*                            cols,
	u32*                            nnz,
	u32**                           col_ptr,
	u32**                           row_idx,
	f32**                           val)
{
	if (!sp) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (sp->format != SPMM_FORMAT_SPARSE_CSC) {
		return SPMM_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
	}

	*rows = sp->rows;
	*cols = sp->cols;
	*nnz = sp->nnz;
	*col_ptr = sp->csc.col_ptr;
	*row_idx = sp->csc.row_idx;
	*val = sp->val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t dn_rm_create(ExecutionContext_t ctx, DnMatDescr_t* dn,
	u32  rows,
	u32  cols,
	f32* val)
{
	if (dn == NULL || *dn != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **dn, (void**)dn) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn)->format = SPMM_FORMAT_DENSE_ROW_MAJOR;

	(*dn)->rows = rows;
	(*dn)->cols = cols;

	const u64 bsize = rows * cols * sizeof *val;
	if (mem_arena_dev_push(&ctx->dev_arena, bsize, (void**)&(*dn)->val) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	cuda_mem_cpy_hd((*dn)->val, val, bsize);

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t dn_rm_get(DnMatDescr* dn,
	u32*                           rows,
	u32*                           cols,
	f32**                          val)
{
	if (!dn) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (dn->format != SPMM_FORMAT_DENSE_ROW_MAJOR) {
		return SPMM_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
	}

	*rows = dn->rows;
	*cols = dn->cols;
	*val = dn->val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t dn_cm_create(ExecutionContext_t ctx, DnMatDescr_t* dn,
	u32  rows,
	u32  cols,
	f32* val)
{
	if (dn == NULL || *dn != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **dn, (void**)dn) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn)->format = SPMM_FORMAT_DENSE_COLUMN_MAJOR;

	(*dn)->rows = rows;
	(*dn)->cols = cols;

	const u64 bsize = rows * cols * sizeof *val;
	if (mem_arena_dev_push(&ctx->dev_arena, bsize, (void**)&(*dn)->val) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	cuda_mem_cpy_hd((*dn)->val, val, bsize);

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t dn_cm_get(DnMatDescr* dn,
	u32*                           rows,
	u32*                           cols,
	f32**                          val)
{
	if (!dn) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (dn->format != SPMM_FORMAT_DENSE_COLUMN_MAJOR) {
		return SPMM_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
	}

	*rows = dn->rows;
	*cols = dn->cols;
	*val = dn->val;

	return SPMM_STATUS_SUCCESS;
}
