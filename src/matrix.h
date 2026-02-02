#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <vector>

enum class SparseMatrixType
{
	CSC,
	CSR
};

struct DLMCHeader
{
	size_t n_rows{}, n_cols{}, nnz{};
};

struct RowMajorHeader
{
	size_t n_rows{}, n_cols{};
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

		mat.row_ptr_bytes = (rows + 1) * sizeof(uint32_t);
		mat.col_idx_bytes = nnz * sizeof(uint32_t);
		mat.val_bytes = nnz * sizeof(float);
		mat.total_bytes = mat.row_ptr_bytes + mat.col_idx_bytes + mat.val_bytes;

		mat.row_ptr = nullptr;
		mat.col_idx = nullptr;
		mat.val = nullptr;
	}

	inline void partition(Matrix& mat, uintptr_t* base_ptr)
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

// std::vector<float> csr_to_row_major(const Csr& mat);
// std::vector<float> csc_to_col_major(const CSC& mat);
float          measure_sparsity(void* s, size_t size);
size_t         calc_sparse_b_size(const size_t n, const size_t nnz);
DLMCHeader     parse_dlmc_header(std::ifstream& file_stream);
RowMajorHeader parse_row_major_header(std::ifstream& file_stream);
void           generate_token_embeddings(void* dst, size_t size);
// Csr                parse_dlmc(void* dst, const std::filesystem::path& filepath);
// CSC                parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
// size_t             calc_max_nnz_per_col(const CSC& csc);
