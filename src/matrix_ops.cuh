#pragma once

#include "common.h"
#include "cuda_fp16.h"

/*
 * C = A*B
 * MxNxK
 * where 
 * A is MxK
 * B is KxN
 * C is MxN
 */

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

// 96 bytes
struct HRPB
{
	std::vector<uint32_t> block_row_ptr{};  // 24 bytes
	std::vector<uint32_t> active_cols{};    // 24 bytes
	std::vector<uint32_t> size_ptr{};       // 24 bytes
	std::vector<Block>    packed_blocks{};  // 24 bytes

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

__host__ SpmmInput deserialize(const std::filesystem::path& filepath);
__host__ void      get_non_zero_col_predicate(PitchedMatrix* pcm_sparse, size_t rows, size_t cols);
