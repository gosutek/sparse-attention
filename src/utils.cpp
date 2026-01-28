#include "utils.h"

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

{
	if (!std::filesystem::exists(filepath) && !std::filesystem::is_regular_file(filepath)) {
		throw std::runtime_error(filepath.string() + " does not exist\n");
	}
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
