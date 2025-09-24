#pragma once

#include <array>
#include <cusparse.h>
#include <filesystem>
#include <string>

constexpr size_t MAX_N_LAYERS = 6;
constexpr size_t MAT_SIZE = 512;
constexpr size_t ALIGNMENT_BYTES = 128;
// 5 = w_q, w_k, w_v, w_o, x
constexpr size_t   MAX_ALLOC = MAX_N_LAYERS * (5 * MAT_SIZE * MAT_SIZE);
constexpr uint16_t BENCHMARKING_DENSE_N_ROWS[] = { 32, 64, 128, 256, 512 };
constexpr uint32_t BENCHMARKING_TOTAL_DENSE_B_SIZE = []() {uint32_t acc = 0; for ( const uint16_t size : BENCHMARKING_DENSE_N_ROWS) { acc += sizeof(float) * size * MAT_SIZE;} return acc; }();
constexpr size_t   BENCHMARKING_ROUNDS = 1;

// TODO: Move these to each kernel's scope.
constexpr size_t N_THREADS = 64;
constexpr size_t WARP_SIZE = 32;

constexpr size_t BK = 256;
constexpr size_t TK = 4;

// NOTE: better
// constexpr size_t N_THREADS = 64;
// constexpr size_t WARP_SIZE = 32;
//
// constexpr size_t BK = 256;
// constexpr size_t TK = 4;

// NOTE: worse
// constexpr size_t N_THREADS = 128;
// constexpr size_t WARP_SIZE = 32;
//
// constexpr size_t BK = 128;
// constexpr size_t TK = 2;
//
constexpr size_t BN = 64;
constexpr size_t TN = 4;

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

struct CuSparse
{
	cusparseHandle_t handle;

	cusparseSpMatDescr_t sparse;
	cusparseDnMatDescr_t dense[5], res[5];

	void*  work_buffer = nullptr;
	size_t work_buffer_size{};

	float alpha = 1.0f, beta = 0.0f;
};

template <typename WeightFormat>
struct SpmmMemHandle
{
	void* data = nullptr;

	float*       d[std::size(BENCHMARKING_DENSE_N_ROWS)] = {};
	WeightFormat s;
	float*       r[std::size(BENCHMARKING_DENSE_N_ROWS)] = {};
};

template <typename WeightFormat>
struct SPMM
{
	std::filesystem::path sparse_path;
	size_t                b_size = 0;

	SpmmMemHandle<WeightFormat> host;
	SpmmMemHandle<WeightFormat> dev;
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
struct MHSAMemHandle
{
	void*  data = nullptr;
	size_t b_size = 0;

	float*                x = nullptr;  // (input_sequence_size, d_m) ~ defaults: (32, 512)
	Weights<WeightFormat> weights;
	MaskFormat            mask;
};

template <typename WeightFormat, typename MaskFormat>
struct MHSA
{
	Config config;
	DLMC   dlmc;

	MHSAMemHandle<WeightFormat, MaskFormat> host;
	MHSAMemHandle<WeightFormat, MaskFormat> dev;
};
