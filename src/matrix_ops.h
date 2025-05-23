#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "cuda_fp16.h"

#define LIKELY(x) __builtin_expect(!!(x), 1)

#define DATA_DIRECTORY "data/"
#define ALIGNMENT 128
#define ROW_PANEL_SIZE 32  // I think this should be the same as TM

#define TM 32
#define TK 16
#define brick_m 16
#define brick_k 4

/*
 * C = A*B
 * MxNxK
 * where 
 * A is MxK
 * B is KxN
 * C is MxN
 */

#ifndef BSR_BLOCK_SIZE
#	define BSR_BLOCK_SIZE 2
#endif

#define THROW_RUNTIME_ERROR(message) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + message)

template <typename T>
std::ostream& operator<<(std::ostream& out_stream, const std::vector<T>& vec)
{
	out_stream << "{";
	for (size_t i = 0; i < vec.size(); ++i) {
		out_stream << vec[i];
		if (i < vec.size() - 1) {
			out_stream << ", ";
		}
	}
	out_stream << "}";
	return out_stream;
}

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& out_stream, const std::array<T, N>& arr)
{
	out_stream << "[";
	for (size_t i = 0; i < arr.size(); ++i) {
		out_stream << arr[i];
		if (i < arr.size() - 1) {
			out_stream << ", ";
		}
	}
	out_stream << "]";
	return out_stream;
}

// TODO: Review structs
struct MatrixHeader
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	size_t row_ptr_bytes;
	size_t col_idx_bytes;
	size_t val_bytes;
	size_t dense_bytes;
};

struct COOElement
{
	uint32_t row, col;

	float val;
};

struct COOMatrix
{
	uint32_t rows, cols, nnz;

	std::vector<COOElement> elements;
};

struct Block
{
	// maybe leave the arrays I only write to uninitialized?
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns{};  // cache-friendly, NOTE: can't this just be a vector, reserved for this size?
	std::array<uint64_t, TK / brick_k + 1>                col_ptr{};
	std::vector<uint64_t>                                 rows{};       // unknown at compile time, paper is wrong
	std::vector<float>                                    nnz_array{};  // unknown at compile time

	uint32_t get_block_size()
	{
		return static_cast<uint32_t>(sizeof(Block) + nnz_array.size() * sizeof(float));
	}

	bool operator==(const Block& other) const
	{
		return (patterns == other.patterns) &&
		       (col_ptr == other.col_ptr) &&
		       (rows == other.rows) &&
		       (nnz_array == other.nnz_array);
	}

	friend std::ostream& operator<<(std::ostream& out_stream, const Block& block)
	{
		out_stream << "\nBlock:\n\t"
				   << "patterns: " << block.patterns << "\n\t"
				   << "col_ptr: " << block.col_ptr << "\n\t"
				   << "rows: " << block.rows << "\n\t"
				   << "nnz_array: " << block.nnz_array << "\n";

		return out_stream;
	}
};

struct HRPB
{
	std::vector<uint32_t> block_row_ptr{};
	std::vector<uint32_t> active_cols{};
	std::vector<uint32_t> size_ptr{};
	std::vector<Block>    packed_blocks{};  // What happens if this has to resize? Do all above elements get moves aswell?

	bool operator==(const HRPB& other) const
	{
		return (block_row_ptr == other.block_row_ptr) &&
		       (active_cols == other.active_cols) &&
		       (size_ptr == other.size_ptr) &&
		       (packed_blocks == other.packed_blocks);
	}

	friend std::ostream& operator<<(std::ostream& out_stream, const HRPB& hrpb)
	{
		out_stream << "HRPB:\n\t"
				   << "block_row_ptr: " << hrpb.block_row_ptr << "\n\t"
				   << "active_cols: " << hrpb.active_cols << "\n\t"
				   << "size_ptr: " << hrpb.size_ptr << "\n\t"
				   << "packed_blocks: " << hrpb.packed_blocks << "\n";

		return out_stream;
	}
};

// TODO: Write description
void      print_matrix_specs(const std::filesystem::path& filepath);
HRPB*     write_hrpb(COOMatrix& mtx, const std::filesystem::path& filepath);
void      write_csr(COOMatrix& mtx, const std::filesystem::path& filepath);
void      convert(const std::filesystem::directory_iterator& target_dir, HRPB* (*conversion_func_ptr)(COOMatrix& mtx, const std::filesystem::path& filepath));
COOMatrix read_mtx(const std::filesystem::path& filepath);
