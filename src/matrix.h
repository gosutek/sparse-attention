#pragma once

#include "common.h"
#include "model.h"

/*
 * C = A*B
 * MxNxK
 * where
 * A is MxK
 * B is KxN
 * C is MxN
 */

enum class BodyType
{
	Encoder,
	Decoder
};

enum class AttentionMechanism
{
	SelfAttention,
	CrossAttention
};

enum class SparseMatrixType
{
	CSC,
	CSR
};

struct DLMCHeader
{
	size_t n_rows{}, n_cols{}, nnz{};
};

/**
 * A tensor of the 4 weight matrices
 * of a single layer
 */
struct Tensor
{
	size_t b_size = 0;

	std::filesystem::path path;

	// Headers for W_Q, W_K, W_V, W_O
	std::array<DLMCHeader, 4> shape;
};

/**
 * Packs whatever is needed for reading the DLMC dataset
 * given a pruning_method and sparsity
 */
struct DLMC
{
	std::string base_path = "data/dlmc/transformer/";
	std::string pruning_method = "l0_regularization/";
	std::string sparsity = "0.5/";

	std::array<const char*, 4> suffixes = { "q.smtx", "k.smtx", "v.smtx", "output_transform.smtx" };

	// NOTE: Will default construct Tensors...
	std::array<Tensor, MAX_N_LAYERS> enc_self_attention_tensors{};
	std::array<Tensor, MAX_N_LAYERS> dec_self_attention_tensors{};

	// std::array<Tensor, MAX_N_LAYERS> enc_cross_attention_tensors{};
	// std::array<Tensor, MAX_N_LAYERS> dec_cross_attention_tensors{};

	DLMC(const std::string& _base_data_path,
		const std::string&  _pruning_method,
		const std::string&  _sparsity) :
		base_path(_base_data_path),
		pruning_method(_pruning_method),
		sparsity(_sparsity) {}
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

	size_t rows{}, cols{}, nnz{};
	size_t row_ptr_size{}, col_idx_size{}, val_size{};

	size_t b_size{};

	CSR() {}

	CSR(size_t _rows, size_t _cols, size_t _nnz) :
		rows(_rows), cols(_cols), nnz(_nnz)
	{
		row_ptr_size = rows + 1;
		col_idx_size = nnz;
		val_size = nnz;

		b_size = row_ptr_size * sizeof(uint32_t) + col_idx_size * sizeof(uint32_t) + val_size * sizeof(float);
	}

	CSR(const CSR& other) = default;
	CSR& operator=(const CSR& other) = default;
	CSR(CSR&& other) = default;
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

	size_t rows{}, cols{}, nnz{};
	size_t col_ptr_size{}, row_idx_size{}, val_size{};

	size_t b_size{};

	CSC() {}

	CSC(size_t _rows, size_t _cols, size_t _nnz) :
		rows(_rows), cols(_cols), nnz(_nnz)
	{
		col_ptr_size = cols + 1;
		row_idx_size = nnz;
		val_size = nnz;

		b_size = col_ptr_size * sizeof(uint32_t) + row_idx_size * sizeof(uint32_t) + val_size * sizeof(float);
	}

	CSC(const CSC& other) :
		rows(other.rows), cols(other.cols), nnz(other.nnz),
		col_ptr_size(other.col_ptr_size), row_idx_size(other.row_idx_size), val_size(other.val_size),
		b_size(other.b_size) {}

	CSC& operator=(const CSC& other) = default;
	CSC(CSC&& other) = default;

	void partition(void* const ptr)
	{
		col_ptr = reinterpret_cast<uint32_t*>(ptr);
		row_idx = col_ptr + col_ptr_size;
		val = reinterpret_cast<float*>(row_idx + row_idx_size);
	}
};

void load_host_csr(
	MHSA<CSR, CSR>     mhsa,
	const Config&      config,
	Weights<CSR>&      weights,
	const std::string& base_data_path,
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism am);

void load_host_csc(
	MHSA<CSC, CSR>&    mhsa,
	const Config&      config,
	Weights<CSC>&      weights,
	const std::string& base_data_path,
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism am);

std::vector<float> csr_to_row_major(const CSR& mat);
std::vector<float> csc_to_col_major(const CSC& mat);
float              measure_sparsity(void* s, size_t size);
size_t             calc_byte_size_compressed_sparse(const size_t n, const size_t nnz);
Tensor             read_tensor(const DLMC& dlmc, const BodyType bt, const AttentionMechanism am, const size_t layer, const SparseMatrixType sparse_matrix_type);
