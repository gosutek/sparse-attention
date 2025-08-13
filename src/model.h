#pragma once

#include "common.h"
#include "matrix.h"

struct Config
{
	size_t n_heads = 1;
	size_t n_layers = 1;
	size_t input_sequence_size = 32;
};

struct Weights
{
	float* x = nullptr;  // (input_sequence_size, d_m) ~ defaults: (32, 512)

	std::array<CSRMatrix, MAX_N_LAYERS> w_q;
	std::array<CSRMatrix, MAX_N_LAYERS> w_k;
	std::array<CSRMatrix, MAX_N_LAYERS> w_v;
	std::array<CSRMatrix, MAX_N_LAYERS> w_o;
};

struct MHSA
{
	Config  config;
	Weights weights;
	void*   host = nullptr;
	size_t  b_size = 0;
};
