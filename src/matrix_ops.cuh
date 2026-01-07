#pragma once

#include "cuda_fp16.h"
#include <array>
#include <filesystem>
#include <vector>

#include "utils.h"

constexpr uint8_t TM = 32;
constexpr uint8_t TK = 16;
constexpr uint8_t brick_m = 16;
constexpr uint8_t brick_k = 4;

constexpr uint8_t ALIGNMENT = 128;
constexpr uint8_t ROW_PANEL_SIZE = 32;  // I think this should be the same as TM

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

/*
 * On the device, with a pointer to another memory chunk on the device
 */
struct PitchedMatrix
{
	__half* data = nullptr;
	size_t  rows{};
	size_t  cols{};
	size_t  pitch{};

	__host__ __device__ PitchedMatrix() {}

	__host__ __device__ PitchedMatrix(__half* _data, size_t _rows, size_t _cols, size_t _pitch) :
		data(_data), rows(_rows), cols(_cols), pitch(_pitch) {}
};

/*
 * Pointers to device on the host
 */
struct SpmmInput
{
	PitchedMatrix* d_pcm_sparse{};
	PitchedMatrix* d_prm_dense{};
	void*          pitched_ptr = nullptr;  // used to free data, d_prm_sparse->data is INVALID (accesing device memory from host)
	size_t         rows{};                 // consumed by the host
	size_t         cols{};                 // consumed by the host
};

// TODO: This need to be converted to __half
// 152 bytes
struct Block
{
	// maybe leave the arrays I only write to uninitialized?
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns{};   // 8 * 8b = 64 bytes | cache-friendly, NOTE: can't this just be a vector, reserved for this size?
	std::array<uint64_t, TK / brick_k + 1>                col_ptr{};    // 5 * 8b = 40 bytes
	std::vector<uint64_t>                                 rows{};       // metadata: should be 3 * 8b = 24b for 64-bit arch, data: unknown at compile time
	std::vector<float>                                    nnz_array{};  // metadata: 24 bytes, data: unknown at compile time

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

__host__ SpmmInput deserialize(const std::filesystem::path& filepath);
__host__ void      get_non_zero_col_predicate(PitchedMatrix* pcm_sparse, size_t rows, size_t cols);
