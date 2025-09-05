#pragma once

#include <array>
#include <filesystem>
#include <string>

constexpr size_t MAX_N_LAYERS = 6;

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

struct SpmmMemoryHandle
{
	void *host{}, *dev{};
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

	BodyType           bt = BodyType::Decoder;
	AttentionMechanism am = AttentionMechanism::SelfAttention;

	std::array<const char*, 4> suffixes = { "q.smtx", "k.smtx", "v.smtx", "output_transform.smtx" };

	// NOTE: Will default construct Tensors...
	std::array<Tensor, MAX_N_LAYERS> enc_self_attention_tensors{};
	std::array<Tensor, MAX_N_LAYERS> dec_self_attention_tensors{};

	// std::array<Tensor, MAX_N_LAYERS> enc_cross_attention_tensors{};
	// std::array<Tensor, MAX_N_LAYERS> dec_cross_attention_tensors{};

	DLMC() {}

	DLMC(const std::string& _base_data_path,
		const std::string&  _pruning_method,
		const std::string&  _sparsity) :
		base_path(_base_data_path),
		pruning_method(_pruning_method),
		sparsity(_sparsity) {}
};

struct Config
{
	size_t n_heads = 1;
	size_t n_layers = 1;
	size_t input_sequence_size = 32;
};

template <typename WeightFormat>
struct Weights
{
	std::array<WeightFormat, MAX_N_LAYERS> w_q;
	std::array<WeightFormat, MAX_N_LAYERS> w_k;
	std::array<WeightFormat, MAX_N_LAYERS> w_v;
	std::array<WeightFormat, MAX_N_LAYERS> w_o;
};

template <typename WeightFormat, typename MaskFormat>
struct MHSA
{
	float* x = nullptr;  // (input_sequence_size, d_m) ~ defaults: (32, 512)

	Config                config;
	DLMC                  dlmc;
	Weights<WeightFormat> weights;
	MaskFormat            mask;

	void* host = nullptr;
	void* dev = nullptr;

	// The total size in bytes for heap allocated objects
	size_t b_size = 0;
};
