#pragma once

#include "../helpers.cuh"
#include "helpers.h"

__global__ void _k_spmm_naive_elemwise_gmem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_ispmm_naive_elemwise_gmem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_spmm_naive_elemwise_smem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_ispmm_naive_elemwise_smem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_spmm_coalesced_nnzwise(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_ispmm_coalesced_nnzwise(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_spmm_coalesced_nnzwise_no_smem(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_ispmm_coalesced_nnzwise_no_smem(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_ispmm_vectorized_nnzwise_regs(
	const f32* __restrict__ dn,
	const u32* __restrict__ col_ptr,
	const u32* __restrict__ row_idx,
	const f32* __restrict__ val,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);

__global__ void _k_spmm_vectorized_nnzwise_regs(
	const u32* __restrict__ row_ptr,
	const u32* __restrict__ col_idx,
	const f32* __restrict__ val,
	const f32* __restrict__ dn,
	const u32 m,
	const u32 k,
	const u32 n,
	f32* __restrict__ res);
