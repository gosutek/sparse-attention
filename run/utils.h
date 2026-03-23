#pragma once

#include <filesystem>
#include <iostream>
#include <random>

#include "helpers.h"

constexpr const f64 ATOL = 1e-7;
constexpr const f64 RTOL = 1e-3;

/*
      * +------------------------------------------------------------------------------+
      * |                                 STRUCTS                                      |
      * +------------------------------------------------------------------------------+
*/

struct Dense
{
	u32 rows;
	u32 cols;

	std::vector<f32> val{};
};

struct CSR
{
	u32 rows;
	u32 cols;
	u32 nnz;

	std::vector<u32> row_ptr{};
	std::vector<u32> col_idx{};
	std::vector<f32> val{};
};

struct CSC
{
	u32 rows;
	u32 cols;
	u32 nnz;

	std::vector<u32> col_ptr;
	std::vector<u32> row_idx;
	std::vector<f32> val;
};

/*
      * +------------------------------------------------------------------------------+
      * |                                 PARSING                                      |
      * +------------------------------------------------------------------------------+
*/

Dense parse_dn_test_case(const std::filesystem::path& path);
CSR   parse_csr_test_case(const std::filesystem::path& path);
CSC   parse_csc_test_case(const std::filesystem::path& path);
CSR   parse_csr_dlmc(const std::filesystem::path& filepath);
CSC   parse_csc_dlmc(const std::filesystem::path& path);

bool verify_res(const f32* const actual, const f32* const expected, size_t n);
f64  calc_cv(const f64 flops, f64& mu, f64& q, const u32 n);

inline f64 mean_f64(const std::vector<f64>& vec)
{
	f32 mean = 0.0f;
	for (u32 i = 0; i < vec.size(); ++i) {
		mean += vec[i];
	}
	return mean / vec.size();
}

inline f32 mean_f32(const std::vector<f32>& vec)
{
	f32 mean = 0.0f;
	for (u32 i = 0; i < vec.size(); ++i) {
		mean += vec[i];
	}
	return mean / vec.size();
}

/*
 * Adapted from:
 * https://github.com/pytorch/pytorch/blob/0d2c383a0607853a3e23de11b0da43a870492c4d/torch/testing/_comparison.py#L610
 */
inline bool comparef(const f32 a, const f32 b)
{
	if (std::isnan(a) || std::isnan(b)) {
		std::cout << "isnan: " << a << " or " << b << std::endl;
		return false;
	}
	if (a == b) {
		return true;
	}

	const f64 abs_diff = std::fabs(a - b);
	const f64 tol = ATOL + RTOL * std::fabs(static_cast<f64>(b));
	if (std::isfinite(abs_diff) && abs_diff <= tol) {
		return true;
	}

	std::cout << "Not close: " << a << " | " << b << std::endl;
	return false;
}

inline bool comparef_test_case(const f32 a, const f32 b)
{
	if (std::isnan(a) || std::isnan(b)) {
		std::cout << "isnan: " << a << " or " << b << std::endl;
		return false;
	}
	if (a == b) {
		std::cout << "Exact: " << a << " | " << b << std::endl;
		return true;
	}

	const f64 abs_diff = std::fabs(a - b);
	const f64 tol = ATOL + RTOL * std::fabs(static_cast<f64>(b));
	if (std::isfinite(abs_diff) && abs_diff <= tol) {
		std::cout << "Tolerance: " << a << " | " << b << std::endl;
		return true;
	}

	std::cout << "Not close: " << a << " | " << b << std::endl;
	return false;
}

// bool verify_res(const f32* const actual, const f32* const expected, size_t n)
// {
//   if
// 	f64 diff = 0.0;
// 	for (size_t i = 0; i < n; ++i) {
// 		diff = std::fabs(actual[i] - expected[i]);
// 		// std::cout << std::format(
// 		// 	"Actual: {}, Expected: {}, Diff: {}, Pos: {}\n", actual[i], expected[i], diff, i);
// 		if (std::isnan(diff) || diff > 0.01) {
// 			std::cout << std::format(
// 				"Values diverge -> Actual: {}, Expected: {} (Diff {:.4f}), pos: {:d}\n",
// 				actual[i], expected[i], diff, i);
// 			return false;
// 		}
// 	}
// 	return true;
// }

template <typename T>
void gen_synth_weights_buffer(void* dst, u64 size)
{
	T* ptr = reinterpret_cast<T*>(dst);

	// INFO: think this is bad, declaring them in each function
	// instead of passing(??)
	std::random_device                  rd;
	std::minstd_rand                    rng(rd());
	std::uniform_real_distribution<f32> uni_real_dist(0.0f, 1.0f);

	for (size_t i = 0; i < size; ++i) {
		ptr[i] = uni_real_dist(rng);
	}
}

template <typename T>
void gen_synth_weights_vec(std::vector<T>& vec, u64 size)
{
	vec.resize(size);
	// INFO: think this is bad, declaring them in each function
	// instead of passing(??)
	std::random_device                  rd;
	std::minstd_rand                    rng(rd());
	std::uniform_real_distribution<f32> uni_real_dist(0.0f, 1.0f);

	for (u32 i = 0; i < size; ++i) {
		vec[i] = uni_real_dist(rng);
	}
}
