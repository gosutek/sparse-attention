#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "cuda_fp16.h"

#define TM 32
#define TK 16
#define brick_m 16
#define brick_k 4

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

struct Block
{
	// maybe leave the arrays I only write to uninitialized?
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns{};  // cache-friendly
	std::array<uint64_t, TK / brick_k + 1>                col_ptr{};
	std::vector<uint64_t>                                 rows{};       // unknown at compile time, paper is wrong
	std::vector<float>                                    nnz_array{};  // unknown at compile time

	uint32_t get_block_size()
	{
		return static_cast<uint32_t>(sizeof(Block) + nnz_array.size() * sizeof(float));
	}
};

struct HRPB
{
	std::vector<uint32_t> block_row_ptr{};
	std::vector<uint32_t> active_cols{};
	std::vector<uint32_t> size_ptr{};
	std::vector<Block>    packed_blocks{};  // What happens if this has to resize? Do all above elements get moves aswell?
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
void      print_matrix_specs(const std::filesystem::path& filepath);
HRPB*     write_hrpb(COOMatrix& mtx, const std::filesystem::path& filepath);
void      write_csr(COOMatrix& mtx, const std::filesystem::path& filepath);
void      convert(const std::filesystem::directory_iterator& target_dir, void (*conversion_func_ptr)(COOMatrix& mtx, const std::filesystem::path& filepath));
COOMatrix read_mtx(const std::filesystem::path& filepath);
