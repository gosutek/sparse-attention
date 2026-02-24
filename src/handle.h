#pragma once

#include "allocator.h"

constexpr size_t MAX_N_LAYERS = 6;
constexpr size_t MAT_SIZE = 512;
constexpr size_t ALIGNMENT_BYTES = 128;
// 5 = w_q, w_k, w_v, w_o, x
constexpr size_t   MAX_ALLOC = MAX_N_LAYERS * (5 * MAT_SIZE * MAT_SIZE);
constexpr uint32_t BENCH_DIMS[] = { 512, 1024 };
constexpr size_t   BENCH_DIMS_BSIZE = []() {size_t acc = 0; for ( const size_t size : BENCH_DIMS) { acc += sizeof(float) * size * MAT_SIZE;} return acc; }();
constexpr size_t   BENCHMARKING_ROUNDS = 1;

// TODO: Move these to each kernel's scope.
constexpr size_t WARP_SIZE = 32;

// struct CuSparse
// {
// 	cusparseHandle_t handle;
//
// 	cusparseSpMatDescr_t sparse;
// 	cusparseDnMatDescr_t dense[5], res[5];
//
// 	void*  work_buffer = nullptr;
// 	size_t work_buffer_size{};
//
// 	float alpha = 1.0f, beta = 0.0f;
// };
//
// template <typename WeightFormat>
// struct SpmmMemHandle
// {
// 	void* data = nullptr;
//
// 	float*       d[std::size(BENCH_DIMS)] = {};
// 	WeightFormat s;
// 	float*       r[std::size(BENCH_DIMS)] = {};
// };
//
// template <typename WeightFormat>
// struct SPMM
// {
// 	std::filesystem::path sparse_path;  // TODO: Remove this crap.
// 	size_t                b_size = 0;
//
// 	SpmmMemHandle<WeightFormat> host;
// 	SpmmMemHandle<WeightFormat> dev;
// };
//
// template <typename WeightFormat>
// struct Weights
// {
// 	std::array<WeightFormat, MAX_N_LAYERS> w_q;
// 	std::array<WeightFormat, MAX_N_LAYERS> w_k;
// 	std::array<WeightFormat, MAX_N_LAYERS> w_v;
// 	std::array<WeightFormat, MAX_N_LAYERS> w_o;
// };
