#include "matrix.h"

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

	if (sp->format != SPARSE_FORMAT_CSR) {
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

	if (sp->format != SPARSE_FORMAT_CSC) {
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

// TODO: Convert to C
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

SpmmStatus_t create_sp_mat_csr(ExecutionContext_t ctx, SpMatDescr_t* sp_mat_descr,
	u32  rows,
	u32  cols,
	u32  nnz,
	u32* row_ptr,
	u32* col_idx,
	f32* val)
{
	if (sp_mat_descr == NULL || *sp_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)ctx, sizeof **sp_mat_descr, (void**)sp_mat_descr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp_mat_descr)->format = SPARSE_FORMAT_CSR;

	(*sp_mat_descr)->rows = rows;
	(*sp_mat_descr)->cols = cols;
	(*sp_mat_descr)->nnz = nnz;

	(*sp_mat_descr)->csr.row_ptr = row_ptr;
	(*sp_mat_descr)->csr.col_idx = col_idx;
	(*sp_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t create_sp_mat_csc(ExecutionContext_t ctx, SpMatDescr_t* sp_mat_descr,
	u32  rows,
	u32  cols,
	u32  nnz,
	u32* col_ptr,
	u32* row_idx,
	f32* val)
{
	if (sp_mat_descr == NULL || *sp_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **sp_mat_descr, (void**)sp_mat_descr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*sp_mat_descr)->format = SPARSE_FORMAT_CSC;

	(*sp_mat_descr)->rows = rows;
	(*sp_mat_descr)->cols = cols;
	(*sp_mat_descr)->nnz = nnz;

	(*sp_mat_descr)->csc.col_ptr = col_ptr;
	(*sp_mat_descr)->csc.row_idx = row_idx;
	(*sp_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t create_dn_mat_row_major(ExecutionContext_t ctx, DnMatDescr_t* dn_mat_descr,
	u32  rows,
	u32  cols,
	f32* val)
{
	if (dn_mat_descr == NULL || *dn_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **dn_mat_descr, (void**)dn_mat_descr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn_mat_descr)->format = DENSE_FORMAT_ROW_MAJOR;

	(*dn_mat_descr)->rows = rows;
	(*dn_mat_descr)->cols = cols;

	(*dn_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}

// TODO: Change this to use the Arena
SpmmStatus_t create_dn_mat_col_major(ExecutionContext_t ctx, DnMatDescr_t* dn_mat_descr,
	u32  rows,
	u32  cols,
	f32* val)
{
	if (dn_mat_descr == NULL || *dn_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (mem_arena_host_push((MemArena*)(ctx), sizeof **dn_mat_descr, (void**)dn_mat_descr) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn_mat_descr)->format = DENSE_FORMAT_COL_MAJOR;

	(*dn_mat_descr)->rows = rows;
	(*dn_mat_descr)->cols = cols;

	(*dn_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}
