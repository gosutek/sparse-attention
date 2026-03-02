#pragma once

#include "../helpers.cuh"

__global__ void _k_spmm_naive_elemwise_gmem_csr(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ d,
	const size_t m,  // sparse rows
	const size_t k,  // sparse cols
	const size_t n,  // dense cols
	float* __restrict__ res);
