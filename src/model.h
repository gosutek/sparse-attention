#pragma once

#include "common.h"
#include "matrix.h"

struct Config
{
	size_t n_heads = 1;
	size_t n_layers = 1;
	size_t input_sequence_size = 32;
};

struct CSRWeights
{
	float* x = nullptr;  // (input_sequence_size, d_m) ~ defaults: (32, 512)

	std::array<CSRMatrix, MAX_N_LAYERS> w_q;
	std::array<CSRMatrix, MAX_N_LAYERS> w_k;
	std::array<CSRMatrix, MAX_N_LAYERS> w_v;
	std::array<CSRMatrix, MAX_N_LAYERS> w_o;
};

struct CSCWeights
{
	float* x = nullptr;  // (input_sequence_size, d_m) ~ defaults: (32, 512)

	std::array<CSCMatrix, MAX_N_LAYERS> w_q;
	std::array<CSCMatrix, MAX_N_LAYERS> w_k;
	std::array<CSCMatrix, MAX_N_LAYERS> w_v;
	std::array<CSCMatrix, MAX_N_LAYERS> w_o;
};

struct CSR_MHSA
{
	Config     config;
	CSRWeights weights;
	void*      host = nullptr;
	size_t     b_size = 0;
};

struct CSC_MHSA
{
	Config     config;
	CSCWeights weights;

	void*  host = nullptr;
	size_t b_size = 0;
};
