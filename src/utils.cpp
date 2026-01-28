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

std::string construct_path(const std::filesystem::path base_path, const BodyType bt, const AttentionMechanism am, const size_t layer)
{
	std::string path = base_path;
	if (bt == BodyType::Encoder) {
		path += "body_encoder_";
	} else {
		path += "body_decoder_";
	}
	path += "layer_" + std::to_string(layer) + "_";

	if (am == AttentionMechanism::SelfAttention) {
		path += "self_attention_multihead_attention_";
	} else {
		path += "encdec_attention_multihead_attention_";
	}
	return path;
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
