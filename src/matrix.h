#if !defined(MATRIX_H)
#define MATRIX_H

#include "spmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocator.h"
#include "cuda_allocator.cuh"
#include "cuda_mem_wrapper.cuh"
#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef enum
	{
		SPMM_FORMAT_SPARSE_CSR = 0,
		SPMM_FORMAT_SPARSE_CSC = 1,
	} FormatSparse_t;

	typedef enum
	{
		SPMM_FORMAT_DENSE_ROW_MAJOR = 0,
		SPMM_FORMAT_DENSE_COLUMN_MAJOR = 1,
	} FormatDense_t;

	typedef struct SpMatDescr
	{
		FormatSparse_t format;

		u32 rows;
		u32 cols;
		u32 nnz;

		union
		{
			struct
			{
				u32* row_ptr;
				u32* col_idx;
			} csr;

			struct
			{
				u32* col_ptr;
				u32* row_idx;
			} csc;
		};

		f32* val;
	} SpMatDescr;

	static inline size_t sp_mat_ptr_count_get(const SpMatDescr* const sp)
	{
		switch (sp->format) {
		case SPMM_FORMAT_SPARSE_CSR:
			return sp->rows + 1;
		case SPMM_FORMAT_SPARSE_CSC:
			return sp->cols + 1;
		default:
			__builtin_unreachable();
		}
	}

	static inline size_t sp_mat_ptr_bytes_get(const SpMatDescr* const sp)
	{
		return sp_mat_ptr_count_get(sp) * (sizeof *(sp->csr.row_ptr));  // sizeof uint might be faster but less flexible when it comes to accepting more types
	}

	static inline size_t sp_mat_idx_count_get(const SpMatDescr* const sp)
	{
		return sp->nnz;
	}

	static inline size_t sp_mat_idx_bytes_get(const SpMatDescr* const sp)
	{
		return sp->nnz * (sizeof *(sp->csr.col_idx));
	}

	static inline size_t sp_mat_val_count_get(const SpMatDescr* const sp)
	{
		return sp->nnz;
	}

	static inline size_t sp_mat_val_bytes_get(const SpMatDescr* const sp)
	{
		return sp->nnz * (sizeof *(sp->val));
	}

	static inline size_t sp_mat_byte_size_get(const SpMatDescr* const sp)
	{
		return sp_mat_ptr_bytes_get(sp) + sp_mat_idx_bytes_get(sp) + sp_mat_val_bytes_get(sp);
	}

	typedef struct DnMatDescr
	{
		FormatDense_t format;

		u32 rows;
		u32 cols;

		f32* val;
	} DnMatDescr;

	static inline u64 dn_mat_bytes_get(const DnMatDescr* const dn)
	{
		return dn->rows * dn->cols * (sizeof *(dn->val));
	}

	// INFO: Probably useless, can just use dn_mat_bytes_get
	static inline u64 spmm_res_mat_bytes_get(const SpMatDescr* const sp, const DnMatDescr* const dn)
	{
		return sp->rows * dn->cols * (sizeof *(sp->val));
	}

	typedef struct
	{
		u32 rows;
		u32 cols;
		u32 nnz;
	} DlmcHeader;

	typedef struct
	{
		u32 rows;
		u32 cols;
	} RowMajorHeader;

	f32 measure_sparsity(void* s, u32 size);
	// size_t             calc_sparse_b_size(const size_t n, const size_t nnz);
	// DlmcHeader         parse_dlmc_header(std::ifstream& file_stream);
	// RowMajorHeader     parse_row_major_header(std::ifstream& file_stream);
	// Csr::Matrix        parse_dlmc(void* dst, const std::filesystem::path& filepath);
	// Csc::Matrix        parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
	// size_t             calc_max_nnz_per_col(const Csc::Matrix& csc);

#if defined(__cplusplus)
}
#endif

#endif  // MATRIX_H
