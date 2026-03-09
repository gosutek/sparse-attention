#pragma once

#include "../helpers.cuh"

__global__ void _k_spmm_naive_elemwise_gmem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res);

__global__ void _k_ispmm_naive_elemwise_gmem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res);

__global__ void _k_ispmm_naive_elemwise_smem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res);
