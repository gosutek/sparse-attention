#include "matrix.h"
#include "spmm.h"

inline size_t sp_mat_ptr_count_get(const SpMatDescr* const sp)
{
	switch (sp->format) {
	case SPARSE_FORMAT_CSR:
		return sp->rows + 1;
	case SPARSE_FORMAT_CSC:
		return sp->cols + 1;
	}
}

inline size_t sp_mat_ptr_bytes_get(const SpMatDescr* const sp)
{
	return sp_mat_ptr_count_get(sp) * (sizeof *(sp->csr.row_ptr));  // sizeof uint might be faster but less flexible when it comes to accepting more types
}

inline size_t sp_mat_idx_count_get(const SpMatDescr* const sp)
{
	return sp->nnz;
}

inline size_t sp_mat_idx_bytes_get(const SpMatDescr* const sp)
{
	return sp->nnz * (sizeof *(sp->csr.col_idx));
}

inline size_t sp_mat_val_count_get(const SpMatDescr* const sp)
{
	return sp->nnz;
}

inline size_t sp_mat_val_bytes_get(const SpMatDescr* const sp)
{
	return sp->nnz * (sizeof *(sp->val));
}

inline size_t sp_mat_byte_size_get(const SpMatDescr* const sp)
{
	return sp_mat_ptr_bytes_get(sp) + sp_mat_idx_bytes_get(sp) + sp_mat_val_bytes_get(sp);
}

inline uint64_t dn_mat_bytes_get(const DnMatDescr* const dn)
{
	return dn->rows * dn->cols * (sizeof *(dn->val));
}

/*
 * Calculates the size of a CSR or CSC matrix in bytes for float values
 * Accounts for non-square matrices
 * n: main dimension's size (cols for CSC, rows for CSR)
 */
static inline size_t get_sparse_byte_size(const size_t n, const size_t nnz)
{
	size_t b_ptr_size = (n + 1) * sizeof(uint32_t);
	size_t b_idx_size = nnz * sizeof(uint32_t);
	size_t b_val_size = nnz * sizeof(float);

	return b_ptr_size + b_idx_size + b_val_size;
}

static SpmmStatus_t get_max_nnz_per_row(size_t* const max_nnz_out, SpMatDescr* const sp_mat_descr)
{
	if (!sp_mat_descr || !max_nnz_out) {
		return SPMM_STATUS_INVALID_VALUE;
	}
	*max_nnz_out = 0;
	const size_t row_ptr_count = sp_mat_ptr_count_get(sp_mat_descr);

	for (size_t i = 0; i < row_ptr_count - 1; ++i) {
		const size_t curr_col_nnz = ((uint32_t*)(sp_mat_descr->csr.row_ptr))[i + 1] - ((uint32_t*)(sp_mat_descr->csr.row_ptr))[i];
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
	memset(&(dn->val), 0, sp->rows * sp->cols * (sizeof(float)));

	for (size_t i = 0; i < sp->rows; ++i) {
		const uint32_t* const row_ptr = sp->csr.row_ptr;
		for (uint32_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
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
	memset(&(dn->val), 0, sp->rows * sp->cols * (sizeof(float)));

	for (size_t i = 0; i < sp->cols; ++i) {
		const uint32_t* const col_ptr = sp->csc.col_ptr;
		for (uint32_t j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
			dn->val[i * sp->rows + sp->csc.row_idx[j]] = sp->val[j];
		}
	}

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr)
{
	return SPMM_STATUS_SUCCESS;
}
// SpmmStatus_t csr_to_csc(Csc::Matrix& mat, const std::vector<uint32_t>& row_ptr_vec, const std::vector<uint32_t>& col_idx_vec)
// {
// 	std::vector<uint32_t> col_count(mat.cols, 0);
// 	for (size_t i = 0; i < mat.nnz; ++i) {
// 		col_count[col_idx_vec[i]]++;
// 	}
//
// 	mat.col_ptr[0] = 0;
// 	for (size_t col = 0; col < mat.cols; ++col) {
// 		mat.col_ptr[col + 1] = mat.col_ptr[col] + col_count[col];
// 	}
//
// 	std::vector<uint32_t> cur_pos(mat.cols);
// 	for (size_t col = 0; col < mat.cols; ++col) {
// 		cur_pos[col] = mat.col_ptr[col];
// 	}
//
// 	for (uint32_t row = 0; row < mat.rows; ++row) {
// 		for (size_t i = row_ptr_vec[row]; i < row_ptr_vec[row + 1]; ++i) {
// 			uint32_t col = col_idx_vec[i];
// 			uint32_t dest_pos = cur_pos[col]++;
// 			mat.row_idx[dest_pos] = row;
// 		}
// 	}
// }

/*
 * Converts mtx from COO to CSR format
 */
// void coo_to_csr(COOMatrix& mtx)
// {
// 	std::vector<int>      row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
// 	std::vector<uint32_t> col_idx(static_cast<size_t>(mtx.nnz));
//
// 	std::vector<float> val(static_cast<size_t>(mtx.nnz));
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
float measure_sparsity(void* s, uint32_t size)
{
	float* ptr = (float*)(s);
	float  nz = .0f;
	for (uint32_t i = 0; i < size; i++) {
		if (ptr[i] == 0)
			nz++;
	}
	return nz / (float)(size);
}

SpmmStatus_t create_sp_mat_csr(SpMatDescr_t* sp_mat_descr,
	uint32_t                                 rows,
	uint32_t                                 cols,
	uint32_t                                 nnz,
	uint32_t*                                row_ptr,
	uint32_t*                                col_idx,
	float*                                   val)
{
	if (sp_mat_descr == NULL || *sp_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	*sp_mat_descr = (SpMatDescr_t)(malloc(sizeof **sp_mat_descr));
	if (*sp_mat_descr == NULL) {
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

SpmmStatus_t create_sp_mat_csc(SpMatDescr_t* sp_mat_descr,
	uint32_t                                 rows,
	uint32_t                                 cols,
	uint32_t                                 nnz,
	uint32_t*                                col_ptr,
	uint32_t*                                row_idx,
	float*                                   val)
{
	if (sp_mat_descr == NULL || *sp_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	*sp_mat_descr = (SpMatDescr_t)(malloc(sizeof **sp_mat_descr));
	if (*sp_mat_descr == NULL) {
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

SpmmStatus_t create_dn_mat_row_major(DnMatDescr_t* dn_mat_descr,
	uint32_t                                       rows,
	uint32_t                                       cols,
	float*                                         val)
{
	if (dn_mat_descr == NULL || *dn_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	*dn_mat_descr = (DnMatDescr_t)(malloc(sizeof **dn_mat_descr));
	if (*dn_mat_descr == NULL) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn_mat_descr)->format = DENSE_FORMAT_ROW_MAJOR;

	(*dn_mat_descr)->rows = rows;
	(*dn_mat_descr)->cols = cols;

	(*dn_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t create_dn_mat_col_major(DnMatDescr_t* dn_mat_descr,
	uint32_t                                       rows,
	uint32_t                                       cols,
	float*                                         val)
{
	if (dn_mat_descr == NULL || *dn_mat_descr != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	*dn_mat_descr = (DnMatDescr_t)(malloc(sizeof **dn_mat_descr));
	if (*dn_mat_descr == NULL) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*dn_mat_descr)->format = DENSE_FORMAT_COL_MAJOR;

	(*dn_mat_descr)->rows = rows;
	(*dn_mat_descr)->cols = cols;

	(*dn_mat_descr)->val = val;

	return SPMM_STATUS_SUCCESS;
}
