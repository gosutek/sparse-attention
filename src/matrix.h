#pragma once

#include "common.h"

#ifndef MAT_SIZE
#	define MAT_SIZE 512
#endif

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
	CSRMatrix weights[1];

	float* embeddings = nullptr;
	size_t embeddings_size = MAT_SIZE * MAT_SIZE;
};

Input  read_input(const std::filesystem::path& filepath);
float* generate_embeddings(size_t size);
