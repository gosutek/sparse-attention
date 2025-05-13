#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "cuda_fp16.h"

// TODO: Review structs
struct MatrixHeader
{
	int32_t rows;
	int32_t cols;
	int64_t nnz;

	size_t row_ptr_bytes;
	size_t col_idx_bytes;
	size_t val_bytes;
	size_t dense_bytes;
};

struct COOElement
{
	int row, col;

	float val;
};

struct COOMatrix
{
	int rows, cols, nnz;

	std::vector<COOElement> elements;
};

/*
 * Generates a dense matrix in column-major format
 * of size rows * cols filled with random values
 */
std::vector<__half> generate_dense(size_t size);

/*
 * Will iterate over all data/ *.mtx matrices
 * and convert them to .bcsr format
 */
void convert(const std::filesystem::directory_iterator& target_dir, void (*conversion_func_ptr)(COOMatrix& mtx, const std::filesystem::path& filepath));

// TODO: Write description
void print_matrix_specs(const std::filesystem::path& filepath);
void write_csr(COOMatrix& mtx, const std::filesystem::path& filepath);
