#pragma once

#include <cstdint>

constexpr uint8_t _CONSTANTS_WARP_SIZE = 32;

__device__ inline bool is_aligned(const void* addr, const size_t alignment_bytes)
{
	return (reinterpret_cast<uintptr_t>(addr) & (alignment_bytes - 1)) == 0;
}

__device__ inline uint8_t align(const void* base, const void* addr, const size_t alignment_bytes)
{
	const uintptr_t offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(base);
	const uintptr_t aligned_offset = (reinterpret_cast<uintptr_t>(offset) + (alignment_bytes - 1)) & ~size_t(alignment_bytes - 1);
	return reinterpret_cast<uintptr_t>(base) + aligned_offset;
}

__device__ inline float get_elem_rm(const float* const a, size_t n_cols, size_t row, size_t col)
{
	return a[row * n_cols + col];
}

__device__ inline float get_elem_cm(const float* const a, size_t n_rows, size_t row, size_t col)
{
	return a[col * n_rows + row];
}

__device__ inline void set_elem_rm(float* const a, size_t n_cols, size_t row, size_t col, float val)
{
	a[row * n_cols + col] = val;
}

__device__ inline void set_elem_cm(float* const a, size_t n_rows, size_t row, size_t col, float val)
{
	a[col * n_rows + row] = val;
}
