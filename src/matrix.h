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

struct DLMCHeader
{
	size_t n_rows, n_cols, nnz;
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
	const std::string& s_pruning_method,
	const std::string& sparsity,
	const std::string& body,
	const std::string& attention_mechanism,
	const int          layer);
float* csr_to_row_major(CSRMatrix& mat);
