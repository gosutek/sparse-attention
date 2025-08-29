#pragma once

#include "common.h"

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
	Weights<WeightFormat> weights;
	MaskFormat            mask;

	void* host = nullptr;
	void* dev = nullptr;

	// The total size in bytes for heap allocated objects
	size_t b_size = 0;
};
