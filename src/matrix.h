#pragma once

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

/**
 * Doesn't and shouldn't own any
 * resources pointed to by row_ptr, col_idx, val
 */
struct CSR
{
	uint32_t* row_ptr = nullptr;
	uint32_t* col_idx = nullptr;
	float*    val = nullptr;

	size_t nrows{}, ncols{}, nnz{};
	size_t row_ptr_size{}, col_idx_size{}, val_size{};

	size_t b_size{};

	CSR() {}

	CSR(size_t rows, size_t cols, size_t nnz) :
		nrows(rows), ncols(cols), nnz(nnz)
	{
		row_ptr_size = nrows + 1;
		col_idx_size = nnz;
		val_size = nnz;

		const size_t row_ptr_b_size = row_ptr_size * sizeof(uint32_t);
		const size_t col_idx_b_size = col_idx_size * sizeof(uint32_t);
		const size_t val_b_size = val_size * sizeof(float);

		b_size = row_ptr_b_size + calc_padding_bytes(row_ptr_b_size, ALIGNMENT_BYTES) +
		         col_idx_b_size + calc_padding_bytes(col_idx_b_size, ALIGNMENT_BYTES) +
		         val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);
	}

	CSR(const CSR& other) = default;

	CSR& operator=(const CSR& other) = default;

	CSR(CSR&& other) = default;

	CSR& operator=(CSR&& other) = default;

	void partition(uintptr_t ptr)
	{
		row_ptr = reinterpret_cast<uint32_t*>(ptr);

		size_t b_size = row_ptr_size * sizeof(uint32_t);
		ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
		col_idx = reinterpret_cast<uint32_t*>(ptr);

		b_size = col_idx_size * sizeof(uint32_t);
		ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
		val = reinterpret_cast<float*>(ptr);
	}
};

/**
 * Doesn't and shouldn't own any
 * resources pointed to by col_ptr, row_idx, val
 */
struct CSC
{
	uint32_t* col_ptr = nullptr;
	uint32_t* row_idx = nullptr;
	float*    val = nullptr;

	size_t nrows{}, ncols{}, nnz{};
	size_t col_ptr_size{}, row_idx_size{}, val_size{};

	size_t b_size{};

	CSC() {}

	CSC(size_t rows, size_t cols, size_t nnz) :
		nrows(rows), ncols(cols), nnz(nnz)
	{
		col_ptr_size = ncols + 1;
		row_idx_size = nnz;
		val_size = nnz;

		const size_t col_ptr_b_size = col_ptr_size * sizeof(uint32_t);
		const size_t row_idx_b_size = row_idx_size * sizeof(uint32_t);
		const size_t val_b_size = val_size * sizeof(float);

		b_size = col_ptr_b_size + calc_padding_bytes(col_ptr_b_size, ALIGNMENT_BYTES) +
		         row_idx_b_size + calc_padding_bytes(row_idx_b_size, ALIGNMENT_BYTES) +
		         val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);
	}

	CSC(const CSC& other) = default;

	CSC& operator=(const CSC& other) = default;

	CSC(CSC&& other) = default;

	CSC& operator=(CSC&& other) = default;

	void partition(uintptr_t ptr)
	{
		col_ptr = reinterpret_cast<uint32_t*>(ptr);

		size_t b_size = col_ptr_size * sizeof(uint32_t);
		ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
		row_idx = reinterpret_cast<uint32_t*>(ptr);

		b_size = row_idx_size * sizeof(uint32_t);
		ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
		val = reinterpret_cast<float*>(ptr);
	}
};

std::vector<float> csr_to_row_major(const CSR& mat);
std::vector<float> csc_to_col_major(const CSC& mat);
float              measure_sparsity(void* s, size_t size);
size_t             calc_sparse_b_size(const size_t n, const size_t nnz);
DLMCHeader         parse_dlmc_header(std::ifstream& file_stream);
RowMajorHeader     parse_row_major_header(std::ifstream& file_stream);
void               generate_token_embeddings(void* dst, size_t size);
CSR                parse_dlmc(void* dst, const std::filesystem::path& filepath);
CSC                parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
size_t             calc_max_nnz_per_col(const CSC& csc);
