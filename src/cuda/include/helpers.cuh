#pragma once

__device__ inline bool      is_aligned(const void* addr, const size_t alignment_bytes);
__device__ inline uintptr_t align(const void* base, const void* addr, const size_t alignment_bytes);
__device__ inline float     get_elem_rm(const float* const a, size_t n_cols, size_t row, size_t col);
__device__ inline float     get_elem_cm(const float* const a, size_t n_rows, size_t row, size_t col);
__device__ inline void      set_elem_rm(float* const a, size_t n_cols, size_t row, size_t col, float val);
__device__ inline void      set_elem_cm(float* const a, size_t n_rows, size_t row, size_t col, float val);
