#pragma once

#include "../helpers.cuh"
#include "helpers.h"

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

__global__ void _k_spmm_naive_elemwise_smem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
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

__global__ void _k_spmm_coalesced_nnzwise(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res);

__global__ void _k_ispmm_coalesced_nnzwise(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res);

__global__ void _k_spmm_coalesced_nnzwise_no_smem(
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const float* __restrict__ dn,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res);

__global__ void _k_ispmm_coalesced_nnzwise_no_smem(
	const float* __restrict__ dn,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res);
