#pragma once

#include "common.h"
#include "matrix.h"

struct Config
{
	size_t n_heads = 1;
	size_t n_layers = 1;
	size_t input_sequence_size = 32;
	size_t vocabulary_size;
	size_t model_dimension;

	inline void parse_header(std::ifstream file_stream)
	{
		std::string token;
		std::string header_line;
		std::getline(file_stream, header_line);

		std::istringstream header_stream(header_line);
		std::getline(header_stream, token, ',');
		vocabulary_size = static_cast<size_t>(std::stoi(token));

		std::getline(header_stream, token, ',');
		model_dimension = static_cast<size_t>(std::stoi(token));
	}

	Config(size_t _n_heads, size_t _n_layers, size_t _input_sequence_size, const std::filesystem::path& filepath) :
		n_heads(_n_heads), n_layers(_n_layers), input_sequence_size(_input_sequence_size)
	{
		if (std::filesystem::exists(filepath / "symbol_modality_33288_512_shared_weights_0_aux.smtx")) {
			parse_header({ filepath / "symbol_modality_33288_512_shared_weights_0_aux.smtx" });
		} else if (std::filesystem::exists(filepath / "symbol_modality_33288_512_shared_.smtx")) {
			parse_header(filepath / "symbol_modality_33288_512_shared_.smtx");
		} else {
			THROW_RUNTIME_ERROR("Undocumented token embeddings table path");
		}
	}
};

struct Weights
{
	CSRMatrix x;  // (vocab, d_m)

	CSRMatrix w_q;
	CSRMatrix w_k;
	CSRMatrix w_v;
	CSRMatrix w_o;
};

struct MHSA
{
	Config  config;
	Weights weights;
	void*   host = nullptr;
	size_t  b_size = 0;

	MHSA(Config _config) :
		config(_config) {}
};
