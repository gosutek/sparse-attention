#pragma once

#include "common.h"

struct MHSA;
struct Weights;
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

struct DLMCHeader
{
	size_t n_rows, n_cols, nnz;
};

struct Tensor
{
	size_t layer;
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

	std::array<const char*, 4> suffixes = { "_q.smtx", "_k.smtx", "_v.smtx", "_output_transform.smtx" };

	std::array<Tensor, MAX_N_LAYERS> enc_self_attention_tensors;

	std::array<Tensor, MAX_N_LAYERS> dec_self_attention_tensors;

	DLMC(const std::string& _base_data_path,
		const std::string&  _pruning_method,
		const std::string&  _sparsity) :
		base_path(_base_data_path),
		pruning_method(_pruning_method),
		sparsity(_sparsity) {}
};

struct CSRMatrix
{
	size_t rows{}, cols{}, nnz{};
	size_t row_ptr_size{}, col_idx_size{}, val_size{};

	uint32_t* row_ptr = nullptr;
	uint32_t* col_idx = nullptr;
	float*    val = nullptr;
};

void read_input(
	MHSA&              mhsa,
	Config&            config,
	Weights&           weights,
	const std::string& base_data_path,
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism attention_mechanism);
std::vector<float> csr_to_row_major(const CSRMatrix& mat);
