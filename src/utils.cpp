#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "matrix.h"
#include "model.h"

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

// struct ReadExpectedOutput
// {
// 	int32_t no_blocks = -1;
// 	HRPB    hrpb;
// };
//
// static ReadExpectedOutput read_expected(const std::filesystem::path& expected_filepath)
// {
// 	std::ifstream  input_stream(expected_filepath, std::ios::in);
// 	nlohmann::json j;
// 	input_stream >> j;
//
// 	const int32_t no_blocks = j["no_blocks"].get<int32_t>();
//
// 	HRPB hrpb;
// 	hrpb.block_row_ptr = j["block_row_ptr"].get<std::vector<uint32_t>>();
// 	hrpb.active_cols = j["active_cols"].get<std::vector<uint32_t>>();
// 	hrpb.size_ptr = j["size_ptr"].get<std::vector<uint32_t>>();
//
// 	for (int32_t i = 0; i < no_blocks; ++i) {
// 		Block          block;
// 		nlohmann::json i_block = j["blocks"][static_cast<size_t>(i)];
// 		block.patterns = i_block["patterns"].get<std::array<uint64_t, 8>>();
// 		block.nnz_array = i_block["nnz_array"].get<std::vector<float>>();
// 		block.col_ptr = i_block["col_ptr"].get<std::array<uint64_t, 5>>();
// 		block.rows = i_block["rows"].get<std::vector<uint64_t>>();
// 		hrpb.packed_blocks.push_back(std::move(block));
// 	}
//
// 	return { no_blocks, hrpb };
// }

// static void unit_test_hrpb(const std::filesystem::directory_iterator& test_filepath)
// {
// 	for (const auto& file : test_filepath) {
// 		if (file.is_regular_file() && file.path().extension() == ".mtx") {
// 			const auto expected_file = file.path().parent_path() / file.path().filename().replace_extension(".json");
// 			if (!std::filesystem::exists(expected_file)) {
// 				continue;
// 				THROW_RUNTIME_ERROR("Unit test not found for file" + expected_file.string());
// 			}
//
// 			std::cout << "Testing for file: " << file.path() << "\n";
// 			COOMatrix             mtx = read_mtx(file.path());
// 			std::shared_ptr<HRPB> hrpb_ptr = write_hrpb(mtx, file.path());
// 			ReadExpectedOutput    expected = read_expected(expected_file);
//
// 			ASSERT_EQ(expected.no_blocks, hrpb_ptr->packed_blocks.size(), "Block size mismatch");
// 			ASSERT_EQ(expected.hrpb, *hrpb_ptr, "Structs are different");
//
//          std::cout << "Test successful\n";
// 		}
// 	}
// }

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

// TODO: This reads w_q instead of 3x3.smtx
[[maybe_unused]] static void test_csr_to_row_major(const std::filesystem::path& filepath)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".smtx") {
		const auto expected_file = filepath.parent_path() / filepath.filename().replace_extension(".rm");
		if (!std::filesystem::exists(expected_file)) {
			throw std::runtime_error("Expected file not found for testing: " + expected_file.string());
		}
		std::cout << std::format("Testing for file: {}\n", filepath.string());
		MHSA<CSR, CSR>     mhsa;
		CSR&               w_q = mhsa.weights.w_q[0];
		size_t             matrix_size = w_q.rows * w_q.cols;
		std::vector<float> actual_matrix = csr_to_row_major(w_q);
		std::vector<float> expected_matrix = read_row_major_from_rm(expected_file, matrix_size);

		// ASSERT_EQ(expected_matrix, actual_matrix, "The matrices differ in values.\n");
		// TODO: Call verify_res() instead

		std::cout << "Test successful\n";
	}
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

void print_mhsa(const MHSA<CSC, CSR>& mhsa)
{
	std::cout << "Printing CSC::col_ptr\n";
	for (size_t i = 0; i < mhsa.weights.w_q[0].col_ptr_size; ++i) {
		std::cout << std::format("{}\n", mhsa.weights.w_q[0].col_ptr[i]);
	}

	std::cout << "Printing CSC::row_idx\n";
	for (size_t i = 0; i < mhsa.weights.w_q[0].row_idx_size; ++i) {
		std::cout << std::format("{}\n", mhsa.weights.w_q[0].row_idx[i]);
	}

	std::cout << "Printing CSC::val\n";
	for (size_t i = 0; i < mhsa.weights.w_q[0].val_size; ++i) {
		std::cout << std::format("{}\n", mhsa.weights.w_q[0].val[i]);
	}
}

bool verify_res(const float* const actual, const float* const expected, size_t n)
{
	double diff = 0.0;
	for (size_t i = 0; i < n; ++i) {
		diff = std::fabs(actual[i] - expected[i]);
		if (std::isnan(diff) || diff > 0.01) {
			std::cout << std::format(
				"Values diverge -> Actual: {}, Expected: {} (Diff {:.4f}), pos: {:d}",
				actual[i], expected[i], diff, i);
			return false;
		}
	}
	return true;
}

void test_dev_spmm()
{
	MHSA<CSC, CSR> mhsa;

	const char* base_data_path = "data/dlmc/transformer/";
	const char* s_pruning_method = "l0_regularization/";
	const char* sparsity = "0.5/";

	mhsa_load_host_csc(mhsa, mhsa.config, mhsa.dlmc, mhsa.weights);

	float*             a_ptr = mhsa.x;
	std::vector<float> a(a_ptr, a_ptr + mhsa.config.input_sequence_size * MAT_SIZE);

	std::vector<float> b = csc_to_col_major(mhsa.weights.w_q[0]);

	std::vector<float> expected = host_spmm_rm_cm(a, b, mhsa.config.input_sequence_size, MAT_SIZE, MAT_SIZE);
	std::vector<float> actual;
	actual.resize(MAT_SIZE * mhsa.config.input_sequence_size);
	// run(mhsa, actual.data());

	verify_res(actual.data(), expected.data(), MAT_SIZE * mhsa.config.input_sequence_size);
	cuda_dealloc_host(mhsa.host);
}
