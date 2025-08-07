#pragma once

#include "common.h"

struct MHSA;
struct Weights;

/*
 * C = A*B
 * MxNxK
 * where 
 * A is MxK
 * B is KxN
 * C is MxN
 */

struct CSRMatrix
{
	size_t rows{}, cols{}, nnz{};
	size_t row_ptr_size{}, col_idx_size{}, val_size{};

	uint32_t* row_ptr = nullptr;
	uint32_t* col_idx = nullptr;
	float*    val = nullptr;
};

/*
 * struct Input constitutes a
 */
struct Input
{
	void*     data;
	uint32_t  b_size;
	CSRMatrix weights[1];

	/* X in col major format */
	float* embeddings = nullptr;
};

void   read_input(MHSA& mhsa, Weights& weights, const std::string& base_data_path, const std::string& s_pruning_method, const std::string& sparsity, const std::string& body, const std::string& attention_mechanism, const int layer);
float* generate_embeddings(size_t size);
float* csr_to_row_major(CSRMatrix& mat);
