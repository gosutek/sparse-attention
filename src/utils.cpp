#include "utils.h"

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

std::vector<std::filesystem::path> collect_rec_input(const std::filesystem::path& path)
{
	uint32_t                                            n_unknown_extention_files{};
	std::vector<std::filesystem::path>                  input_files;
	const std::filesystem::recursive_directory_iterator rec_dir_iter(path);

	for (const std::filesystem::path& path : rec_dir_iter) {
		if (std::filesystem::is_regular_file(path) && path.extension() == ".smtx") {
			input_files.push_back(path);
		} else {
			n_unknown_extention_files++;
		}
	}
	std::cout << std::format("Found in directory '{}':\n", path.string())
			  << std::format("\t- {} '.smtx' file(s)\n", input_files.size())
			  << std::format("\t- {} unsupported file(s)\n", n_unknown_extention_files);

	return input_files;
}

/*
 * a(m, k)
 * b(k, n)
 * c(m, n)
 * Expects b to be in column-major
 */
[[maybe_unused]] static std::vector<float> host_spmm_rm_cm(const std::vector<float>& a, const std::vector<float>& b, size_t m, size_t k, size_t n)
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

[[maybe_unused]] static std::vector<float> host_spmm_rm_rm(std::vector<float> a, std::vector<float> b, size_t m, size_t k, size_t n)
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
