#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

/*
      * +------------------------------------------------------------------------------+
      * |                                 STRUCTS                                      |
      * +------------------------------------------------------------------------------+
*/

struct CSR
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	std::vector<uint32_t> row_ptr{};
	std::vector<uint32_t> col_idx{};
	std::vector<float>    val{};
};

struct CSC
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	std::vector<uint32_t> col_ptr;
	std::vector<uint32_t> row_idx;
	std::vector<float>    val;
};

/*
      * +------------------------------------------------------------------------------+
      * |                                 PARSING                                      |
      * +------------------------------------------------------------------------------+
*/

CSR parse_csr_test_case(const std::filesystem::path& path);
CSC parse_csc_test_case(const std::filesystem::path& path);
CSR parse_csr_dlmc(const std::filesystem::path& filepath);

bool verify_res(const float* const actual, const float* const expected, size_t n);

/*
 * Adapted from:
 * https://github.com/pytorch/pytorch/blob/0d2c383a0607853a3e23de11b0da43a870492c4d/torch/testing/_comparison.py#L610
 */
inline bool float_compare(const float a, const float b)
{
	if (std::isnan(a) || std::isnan(b)) {
		return false;
	}
	if (a == b) {
		return true;
	}

	double abs_diff = std::fabs(a - b);
	if (abs_diff > 0.01) {
		return false;
	}

	return true;
}

// bool verify_res(const float* const actual, const float* const expected, size_t n)
// {
//   if
// 	double diff = 0.0;
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
void gen_synth_weights_buffer(void* dst, uint64_t size)
{
	T* ptr = reinterpret_cast<T*>(dst);

	// INFO: think this is bad, declaring them in each function
	// instead of passing(??)
	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	for (size_t i = 0; i < size; ++i) {
		ptr[i] = uni_real_dist(rng);
	}
}

template <typename T>
void gen_synth_weights_vec(std::vector<T>& vec, uint64_t size)
{
	vec.reserve(size);
	// INFO: think this is bad, declaring them in each function
	// instead of passing(??)
	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	for (uint32_t i = 0; i < size; ++i) {
		vec.push_back(uni_real_dist(rng));
	}
}
