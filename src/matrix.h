#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

enum sparseFormat_t
{
	SPARSE_FORMAT_CSR = 1,
	SPARSE_FORMAT_CSC = 2,
};

enum indexType_t
{
	INDEX_TYPE_16U = 1,
	INDEX_TYPE_32U = 2,
	INDEX_TYPE_64U = 3,
};

enum dataType_t
{
	DATA_TYPE_F32 = 1,
};

struct SpMatDescr
{
	sparseFormat_t format;
	indexType_t    index_type;
	dataType_t     data_type;

	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	union
	{
		struct
		{
			void* row_ptr;
			void* col_idx;

			size_t row_ptr_cnt;
			size_t col_idx_cnt;

			size_t row_ptr_bytes;
			size_t col_idx_bytes;
		} csr;

		struct
		{
			void* col_ptr;
			void* row_idx;

			size_t col_ptr_cnt;
			size_t row_idx_cnt;

			size_t col_ptr_bytes;
			size_t row_idx_bytes;
		} csc;
	};

	void* values;

	size_t val_ptr_cnt;
	size_t val_ptr_bytes;

	size_t total_bytes;
};

typedef SpMatDescr* SpMatDescr_t;

struct DlmcHeader
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;
};

struct RowMajorHeader
{
	uint32_t rows;
	uint32_t cols;
};

namespace Csr
{
	struct Matrix
	{
		uint32_t rows;
		uint32_t cols;
		uint32_t nnz;

		uint32_t* row_ptr;
		uint32_t* col_idx;
		float*    val;

		size_t row_ptr_count;
		size_t col_idx_count;
		size_t val_count;

		size_t row_ptr_bytes;
		size_t col_idx_bytes;
		size_t val_bytes;
		size_t total_bytes;
	};

	inline void init(Matrix& mat, uint32_t rows, uint32_t cols, uint32_t nnz)
	{
		mat.rows = rows;
		mat.cols = cols;
		mat.nnz = nnz;

		mat.row_ptr_count = rows + 1;
		mat.col_idx_count = nnz;
		mat.val_count = nnz;

		mat.row_ptr_bytes = (rows + 1) * sizeof(uint32_t);
		mat.col_idx_bytes = nnz * sizeof(uint32_t);
		mat.val_bytes = nnz * sizeof(float);
		mat.total_bytes = mat.row_ptr_bytes + mat.col_idx_bytes + mat.val_bytes;

		mat.row_ptr = nullptr;
		mat.col_idx = nullptr;
		mat.val = nullptr;
	}

	inline void partition(Matrix& mat, uintptr_t base_ptr)
	{
		mat.row_ptr = reinterpret_cast<uint32_t*>(base_ptr);

		base_ptr += mat.row_ptr_bytes;
		mat.col_idx = reinterpret_cast<uint32_t*>(base_ptr);

		base_ptr += mat.col_idx_bytes;
		mat.val = reinterpret_cast<float*>(base_ptr);
	}
}  // namespace Csr

namespace Csc
{
	struct Matrix
	{
		uint32_t rows;
		uint32_t cols;
		uint32_t nnz;

		uint32_t* col_ptr;
		uint32_t* row_idx;
		float*    val;

		size_t col_ptr_count;
		size_t row_idx_count;
		size_t val_count;

		size_t col_ptr_bytes;
		size_t row_idx_bytes;
		size_t val_bytes;
		size_t total_bytes;
	};

	inline void init(Matrix& mat, uint32_t rows, uint32_t cols, uint32_t nnz)
	{
		mat.rows = rows;
		mat.cols = cols;
		mat.nnz = nnz;

		mat.col_ptr_count = cols + 1;
		mat.row_idx_count = nnz;
		mat.val_count = nnz;

		mat.col_ptr_bytes = (cols + 1) * sizeof(uint32_t);
		mat.row_idx_bytes = nnz * sizeof(uint32_t);
		mat.val_bytes = nnz * sizeof(float);
		mat.total_bytes = mat.col_ptr_bytes + mat.row_idx_bytes + mat.val_bytes;
	}

	inline void partition(Matrix& mat, uintptr_t base_ptr)
	{
		mat.col_ptr = reinterpret_cast<uint32_t*>(base_ptr);

		base_ptr += mat.col_ptr_bytes;
		mat.row_idx = reinterpret_cast<uint32_t*>(base_ptr);

		base_ptr += mat.row_idx_bytes;
		mat.val = reinterpret_cast<float*>(base_ptr);
	}
}  // namespace Csc

void create_sp_mat_csr(SpMatDescr_t& sp_mat_descr,
	uint32_t                         rows,
	uint32_t                         cols,
	uint32_t                         nnz,
	void*                            row_ptr,
	void*                            col_idx,
	void*                            values,
	indexType_t                      index_type,
	dataType_t                       val_type);

void create_sp_mat_csc(SpMatDescr_t& sp_mat_descr,
	uint32_t                         rows,
	uint32_t                         cols,
	uint32_t                         nnz,
	void*                            col_ptr,
	void*                            row_idx,
	void*                            values,
	indexType_t                      index_type,
	dataType_t                       val_type);

std::vector<float> csr_to_row_major(const Csr::Matrix& mat);
std::vector<float> csc_to_col_major(const Csc::Matrix& mat);
float              measure_sparsity(void* s, size_t size);
size_t             calc_sparse_b_size(const size_t n, const size_t nnz);
DlmcHeader         parse_dlmc_header(std::ifstream& file_stream);
RowMajorHeader     parse_row_major_header(std::ifstream& file_stream);
void               generate_token_embeddings(void* dst, size_t size);
Csr::Matrix        parse_dlmc(void* dst, const std::filesystem::path& filepath);
Csc::Matrix        parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
size_t             calc_max_nnz_per_col(const Csc::Matrix& csc);
