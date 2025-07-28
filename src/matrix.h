#pragma once

#include "common.h"

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
	uint32_t rows{}, cols{}, nnz{}, row_ptr_size{}, col_idx_size{}, val_size{};

	uint32_t* row_ptr = nullptr;
	uint32_t* col_idx = nullptr;
	float*    val = nullptr;
};

struct Input
{
	void*     data;
	uint32_t  b_size;
	CSRMatrix weights[1];

	/* X in col major format */
	float* embeddings = nullptr;
};

Input  read_input(const std::filesystem::path& filepath);
float* generate_embeddings(size_t size);
float* csr_to_row_major(CSRMatrix& mat);
