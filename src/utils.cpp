#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "matrix.h"

#ifndef MAT_SIZE
#	define MAT_SIZE 512
#endif

// TODO: expected == actual is garbage
#define ASSERT_EQ(expected, actual, message)                           \
	do {                                                               \
		if (!((expected) == (actual))) {                               \
			std::cerr << "Assertion failed: " << message << std::endl; \
			std::cerr << " Expected: " << expected << std::endl;       \
			std::cerr << " Actual:   " << actual << std::endl;         \
			exit(EXIT_FAILURE);                                        \
		}                                                              \
	} while (0)

void run(MHSA<CSC, CSR>& mhsa, float* res);
void cuda_dealloc_host(void* ptr);

static std::vector<float> read_row_major_from_rm(const std::filesystem::path& filepath, size_t size)
{
	std::vector<float> res;
	res.reserve(size);

	std::ifstream file_stream(filepath, std::ios_base::in);
	if (!file_stream) {
		throw std::runtime_error("Failed to open file:" + filepath.string());
	}
	float tmp;
	while (file_stream >> tmp) {
		res.push_back(tmp);
	}
	return res;
}

/*
 * a(m, k)
 * b(k, n)
 * c(m, n)
 * Expects b to be in column-major
 */
static std::vector<float> host_spmm_rm_cm(const std::vector<float>& a, const std::vector<float>& b, size_t m, size_t k, size_t n)
{
	std::vector<float> res;
	res.reserve(m * n);

	for (size_t a_row = 0; a_row < m; ++a_row) {
		for (size_t b_col = 0; b_col < n; ++b_col) {
			float acc = 0;
			for (size_t i = 0; i < k; ++i) {
				acc += a[a_row * k + i] * b[b_col * k + i];
			}
			res.push_back(acc);
		}
	}

	return res;
}

static std::vector<float> host_spmm_rm_rm(std::vector<float> a, std::vector<float> b, size_t m, size_t k, size_t n)
{
	std::vector<float> res;
	res.reserve(m * n);

	for (size_t a_row = 0; a_row < m; ++a_row) {
		for (size_t b_col = 0; b_col < n; ++b_col) {
			float acc = 0;
			for (size_t i = 0; i < k; ++i) {
				acc += a[a_row * k + i] * b[i * k + b_col];
			}
			res.push_back(acc);
		}
	}

	return res;
}

[[maybe_unused]] static void test_host_spmm_rm_cm(const std::filesystem::path& filepath, size_t m, size_t k, size_t n)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".rm") {
		const auto a_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_a.rm"));
		const auto b_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_b.cm"));
		if (!std::filesystem::exists(a_matrix_file)) {
			throw std::runtime_error("Expected file not found for testing: " + a_matrix_file.string());
		}
		if (!std::filesystem::exists(b_matrix_file)) {
			throw std::runtime_error("Expected file not found for testing: " + b_matrix_file.string());
		}
		std::cout << std::format("Testing 'host_spmm' with file: {}\n", filepath.string());
		// WARN: change hardcoded values
		std::vector<float> a = read_row_major_from_rm(a_matrix_file, m * k);
		std::vector<float> b = read_row_major_from_rm(b_matrix_file, k * n);
		std::vector<float> actual = host_spmm_rm_cm(a, b, m, k, n);
		std::vector<float> expected = read_row_major_from_rm(filepath, m * n);

		// ASSERT_EQ(expected, actual, "The matrices differ in values.\n");
		// TODO: Call verify_res() instead

		std::cout << "Test successful\n";
	}
}

[[maybe_unused]] static void test_host_spmm_rm_rm(const std::filesystem::path& filepath, size_t m, size_t k, size_t n)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".rm") {
		const auto a_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_a.rm"));
		const auto b_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_b.rm"));
		if (!std::filesystem::exists(a_matrix_file)) {
			throw std::runtime_error("Expected file not found for testing: " + a_matrix_file.string());
		}
		if (!std::filesystem::exists(b_matrix_file)) {
			throw std::runtime_error("Expected file not found for testing: " + b_matrix_file.string());
		}
		std::cout << std::format("Testing 'host_spmm' with file: {}\n", filepath.string());
		std::vector<float> a = read_row_major_from_rm(a_matrix_file, m * k);
		std::vector<float> b = read_row_major_from_rm(b_matrix_file, k * n);
		std::vector<float> actual = host_spmm_rm_rm(a, b, m, k, n);
		std::vector<float> expected = read_row_major_from_rm(filepath, m * n);

		// ASSERT_EQ(expected, actual, "The matrices differ in values.\n");
		// TODO: Call verify_res() instead

		std::cout << "Test successful\n";
	}
}

bool verify_res(const float* const actual, const float* const expected, size_t n)
{
	double diff = 0.0;
	for (size_t i = 0; i < n; ++i) {
		diff = std::fabs(actual[i] - expected[i]);
		// std::cout << std::format(
		// 	"Actual: {}, Expected: {}, Diff: {}, Pos: {}\n", actual[i], expected[i], diff, i);
		if (std::isnan(diff) || diff > 0.01) {
			std::cout << std::format(
				"Values diverge -> Actual: {}, Expected: {} (Diff {:.4f}), pos: {:d}\n",
				actual[i], expected[i], diff, i);
			return false;
		}
	}
	return true;
}
