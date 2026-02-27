#if !defined(MATRIX_H)
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spmm.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef enum
	{
		SPARSE_FORMAT_CSR = 0,
		SPARSE_FORMAT_CSC = 1,
	} SparseFormat_t;

	typedef enum
	{
		DENSE_FORMAT_ROW_MAJOR = 0,
		DENSE_FORMAT_COL_MAJOR = 1,
	} DenseFormat_t;

	typedef struct SpMatDescr
	{
		SparseFormat_t format;

		uint32_t rows;
		uint32_t cols;
		uint32_t nnz;

		union
		{
			struct
			{
				uint32_t* row_ptr;
				uint32_t* col_idx;
			} csr;

			struct
			{
				uint32_t* col_ptr;
				uint32_t* row_idx;
			} csc;
		};

		float* val;
	} SpMatDescr;

	static inline size_t sp_mat_ptr_count_get(const SpMatDescr* const sp)
	{
		switch (sp->format) {
		case SPARSE_FORMAT_CSR:
			return sp->rows + 1;
		case SPARSE_FORMAT_CSC:
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
		DenseFormat_t format;

		uint32_t rows;
		uint32_t cols;

		float* val;
	} DnMatDescr;

	static inline uint64_t dn_mat_bytes_get(const DnMatDescr* const dn)
	{
		return dn->rows * dn->cols * (sizeof *(dn->val));
	}

	static inline uint64_t spmm_res_mat_bytes_get(const SpMatDescr* const sp, const DnMatDescr* const dn)
	{
		return sp->rows * dn->cols * (sizeof *(sp->val));
	}

	typedef struct
	{
		uint32_t rows;
		uint32_t cols;
		uint32_t nnz;
	} DlmcHeader;

	typedef struct
	{
		uint32_t rows;
		uint32_t cols;
	} RowMajorHeader;

	// namespace Csr
	// {
	// 	struct Matrix
	// 	{
	// 		uint32_t rows;
	// 		uint32_t cols;
	// 		uint32_t nnz;
	//
	// 		uint32_t* row_ptr;
	// 		uint32_t* col_idx;
	// 		float*    val;
	//
	// 		size_t row_ptr_count;
	// 		size_t col_idx_count;
	// 		size_t val_count;
	//
	// 		size_t row_ptr_bytes;
	// 		size_t col_idx_bytes;
	// 		size_t val_bytes;
	// 		size_t total_bytes;
	// 	};
	//
	// 	inline void init(Matrix& mat, uint32_t rows, uint32_t cols, uint32_t nnz)
	// 	{
	// 		mat.rows = rows;
	// 		mat.cols = cols;
	// 		mat.nnz = nnz;
	//
	// 		mat.row_ptr_count = rows + 1;
	// 		mat.col_idx_count = nnz;
	// 		mat.val_count = nnz;
	//
	// 		mat.row_ptr_bytes = (rows + 1) * sizeof(uint32_t);
	// 		mat.col_idx_bytes = nnz * sizeof(uint32_t);
	// 		mat.val_bytes = nnz * sizeof(float);
	// 		mat.total_bytes = mat.row_ptr_bytes + mat.col_idx_bytes + mat.val_bytes;
	//
	// 		mat.row_ptr = nullptr;
	// 		mat.col_idx = nullptr;
	// 		mat.val = nullptr;
	// 	}
	//
	// 	inline void partition(Matrix& mat, uintptr_t base_ptr)
	// 	{
	// 		mat.row_ptr = reinterpret_cast<uint32_t*>(base_ptr);
	//
	// 		base_ptr += mat.row_ptr_bytes;
	// 		mat.col_idx = reinterpret_cast<uint32_t*>(base_ptr);
	//
	// 		base_ptr += mat.col_idx_bytes;
	// 		mat.val = reinterpret_cast<float*>(base_ptr);
	// 	}
	// }  // namespace Csr
	//
	// namespace Csc
	// {
	// 	struct Matrix
	// 	{
	// 		uint32_t rows;
	// 		uint32_t cols;
	// 		uint32_t nnz;
	//
	// 		uint32_t* col_ptr;
	// 		uint32_t* row_idx;
	// 		float*    val;
	//
	// 		size_t col_ptr_count;
	// 		size_t row_idx_count;
	// 		size_t val_count;
	//
	// 		size_t col_ptr_bytes;
	// 		size_t row_idx_bytes;
	// 		size_t val_bytes;
	// 		size_t total_bytes;
	// 	};
	//
	// 	inline void init(Matrix& mat, uint32_t rows, uint32_t cols, uint32_t nnz)
	// 	{
	// 		mat.rows = rows;
	// 		mat.cols = cols;
	// 		mat.nnz = nnz;
	//
	// 		mat.col_ptr_count = cols + 1;
	// 		mat.row_idx_count = nnz;
	// 		mat.val_count = nnz;
	//
	// 		mat.col_ptr_bytes = (cols + 1) * sizeof(uint32_t);
	// 		mat.row_idx_bytes = nnz * sizeof(uint32_t);
	// 		mat.val_bytes = nnz * sizeof(float);
	// 		mat.total_bytes = mat.col_ptr_bytes + mat.row_idx_bytes + mat.val_bytes;
	// 	}
	//
	// 	inline void partition(Matrix& mat, uintptr_t base_ptr)
	// 	{
	// 		mat.col_ptr = reinterpret_cast<uint32_t*>(base_ptr);
	//
	// 		base_ptr += mat.col_ptr_bytes;
	// 		mat.row_idx = reinterpret_cast<uint32_t*>(base_ptr);
	//
	// 		base_ptr += mat.row_idx_bytes;
	// 		mat.val = reinterpret_cast<float*>(base_ptr);
	// 	}
	// }  // namespace Csc

	// std::vector<float> csr_to_row_major(const Csr::Matrix& mat);
	// std::vector<float> csc_to_col_major(const Csc::Matrix& mat);
	float measure_sparsity(void* s, uint32_t size);
	// size_t             calc_sparse_b_size(const size_t n, const size_t nnz);
	// DlmcHeader         parse_dlmc_header(std::ifstream& file_stream);
	// RowMajorHeader     parse_row_major_header(std::ifstream& file_stream);
	// void               generate_token_embeddings(void* dst, size_t size);
	// Csr::Matrix        parse_dlmc(void* dst, const std::filesystem::path& filepath);
	// Csc::Matrix        parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
	// size_t             calc_max_nnz_per_col(const Csc::Matrix& csc);

#if defined(__cplusplus)
}
#endif

#endif  // MATRIX_H
