#pragma once

#include "common.h"

struct CSR_MHSA;
struct CSC_MHSA;
struct CSRWeights;
struct CSCWeights;
struct Config;

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

struct Tensor
{
	size_t layer{};
	size_t b_size = 0;

	BodyType           bt;
	AttentionMechanism am;

	std::filesystem::path path;

	std::array<DLMCHeader, 4> shape;
};

struct DLMC
{
	std::string base_path = "data/dlmc/transformer/";
	std::string pruning_method = "l0_regularization/";
	std::string sparsity = "0.5/";

	std::array<const char*, 4> suffixes = { "q.smtx", "k.smtx", "v.smtx", "output_transform.smtx" };

	std::array<Tensor, MAX_N_LAYERS> enc_self_attention_tensors{};

	std::array<Tensor, MAX_N_LAYERS> dec_self_attention_tensors{};

	DLMC(const std::string& _base_data_path,
		const std::string&  _pruning_method,
		const std::string&  _sparsity) :
		base_path(_base_data_path),
		pruning_method(_pruning_method),
		sparsity(_sparsity) {}
};

struct CSRMatrix
{
	uint32_t* row_ptr = nullptr;
	uint32_t* col_idx = nullptr;
	float*    val = nullptr;

	size_t rows{}, cols{}, nnz{};
	size_t row_ptr_size{}, col_idx_size{}, val_size{};
};

struct CSCMatrix
{
	uint32_t* col_ptr = nullptr;
	uint32_t* row_idx = nullptr;
	float*    val = nullptr;

	size_t rows{}, cols{}, nnz{};
	size_t col_ptr_size{}, row_idx_size{}, val_size{};

	size_t b_size{};

	CSCMatrix() {}

	CSCMatrix(size_t _rows, size_t _cols, size_t _nnz) :
		rows(_rows), cols(_cols), nnz(_nnz)
	{
		col_ptr_size = cols + 1;
		row_idx_size = nnz;
		val_size = nnz;

		b_size = col_ptr_size * sizeof(uint32_t) + row_idx_size * sizeof(uint32_t) + val_size * sizeof(float);
	}

	CSCMatrix(const CSCMatrix& other) :
		rows(other.rows), cols(other.cols), nnz(other.nnz),
		col_ptr_size(other.col_ptr_size), row_idx_size(other.row_idx_size), val_size(other.val_size),
		b_size(other.b_size) {}

	CSCMatrix& operator=(const CSCMatrix& other) = default;

public:
	void partition(void* const ptr)
	{
		col_ptr = reinterpret_cast<uint32_t*>(ptr);
		row_idx = col_ptr + col_ptr_size;
		val = reinterpret_cast<float*>(row_idx + row_idx_size);
	}
};

void load_host_csr(
	CSR_MHSA&          mhsa,
	const Config&      config,
	CSRWeights&        weights,
	const std::string& base_data_path,
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism am);

void load_host_csc(
	CSC_MHSA&          mhsa,
	const Config&      config,
	CSCWeights&        weights,
	const std::string& base_data_path,
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism am);

std::vector<float> csr_to_row_major(const CSRMatrix& mat);
std::vector<float> csc_to_col_major(const CSCMatrix& mat);
float              measure_sparsity(void* s, size_t size);
size_t             calc_byte_size_compressed_sparse(const size_t n, const size_t nnz);
