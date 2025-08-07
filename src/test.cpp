#include "../extern/json.hpp"

#include "common.h"
#include "matrix.h"
#include "model.h"

#ifndef MAT_SIZE
#	define MAT_SIZE 512
#endif

#define ASSERT_EQ(expected, actual, message)                           \
	do {                                                               \
		if (!((expected) == (actual))) {                               \
			std::cerr << "Assertion failed: " << message << std::endl; \
			std::cerr << " Expected: " << expected << std::endl;       \
			std::cerr << " Actual:   " << actual << std::endl;         \
			exit(EXIT_FAILURE);                                        \
		}                                                              \
	} while (0)

void run(Input input);
void cuda_dealloc_host(void* ptr);

MHSA default_run()
{
	MHSA mhsa;

	std::string base_data_path = "data/dlmc/transformer/";
	std::string s_pruning_method = "l0_regularization/";
	std::string sparsity = "0.5/";
	std::string body = "body_decoder_";
	std::string attention_mechanism = "self_attention_multihead_attention_";
	int         n_layers = 0;

	read_input(mhsa, mhsa.weights, base_data_path, s_pruning_method, sparsity, body, attention_mechanism, n_layers);

	return mhsa;
}

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
// 			printf("Test successful\n");
// 		}
// 	}
// }

static std::vector<float> read_row_major_from_rm(const std::filesystem::path& filepath, size_t size)
{
	std::vector<float> res;
	res.reserve(size);

	std::ifstream file_stream(filepath, std::ios_base::in);
	if (!file_stream) {
		THROW_RUNTIME_ERROR("Failed to open file.\n");
	}
	float tmp;
	while (file_stream >> tmp) {
		res.push_back(tmp);
	}
	return res;
}

[[maybe_unused]] static void test_csr_to_row_major(const std::filesystem::path& filepath)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".smtx") {
		const auto expected_file = filepath.parent_path() / filepath.filename().replace_extension(".rm");
		if (!std::filesystem::exists(expected_file)) {
			THROW_RUNTIME_ERROR("Expected file not found for testing: " + expected_file.string());
		}
		std::cout << "Testing for file: " << filepath << "\n";
		Input              input = read_input(filepath);
		size_t             matrix_size = input.weights[0].rows * input.weights[0].cols;
		float*             actual_matrix_ptr = csr_to_row_major(input.weights[0]);
		std::vector<float> actual_matrix;
		actual_matrix.reserve(matrix_size);
		std::vector<float> expected_matrix = read_row_major_from_rm(expected_file, matrix_size);

		for (size_t i = 0; i < matrix_size; ++i) {
			actual_matrix.push_back(actual_matrix_ptr[i]);
		}
		ASSERT_EQ(expected_matrix, actual_matrix, "The matrices differ in values.\n");

		printf("Test successful\n");

		std::free(actual_matrix_ptr);
	}
}

static std::vector<float> host_spmm(std::vector<float> a, std::vector<float> b, size_t rows, size_t cols)
{
	std::vector<float> res;
	res.reserve(rows * cols);

	for (size_t a_row = 0; a_row < rows; ++a_row) {
		for (size_t b_col = 0; b_col < cols; ++b_col) {
			float acc = 0;
			for (size_t i = 0; i < cols; ++i) {
				acc += a[a_row * cols + i] * b[b_col * rows + i];
			}
			res.push_back(acc);
		}
	}

	return res;
}

[[maybe_unused]] static void test_host_spmm(const std::filesystem::path& filepath)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".rm") {
		const auto a_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_a.rm"));
		const auto b_matrix_file = filepath.parent_path() / filepath.stem().replace_filename(filepath.stem().string().append("_b.rm"));
		if (!std::filesystem::exists(a_matrix_file)) {
			THROW_RUNTIME_ERROR("Expected file not found for testing: " + a_matrix_file.string());
		}
		if (!std::filesystem::exists(b_matrix_file)) {
			THROW_RUNTIME_ERROR("Expected file not found for testing: " + b_matrix_file.string());
		}
		std::cout << "Testing 'host_spmm' with file: " << filepath << "\n";
		// WARN: change hardcoded 9
		std::vector<float> a = read_row_major_from_rm(a_matrix_file, 9);
		std::vector<float> b = read_row_major_from_rm(b_matrix_file, 9);
		std::vector<float> actual = host_spmm(a, b, 3, 3);
		std::vector<float> expected = read_row_major_from_rm(filepath, 9);

		ASSERT_EQ(expected, actual, "The matrices differ in values.\n");

		printf("Test successful\n");
	}
	std::ifstream file_stream(filepath, std::ios_base::in);
}

static void test_dev_spmm(const std::filesystem::path& filepath)
{
	if (std::filesystem::is_regular_file(filepath) && filepath.extension() == ".smtx") {
		std::cout << "Testing 'dev_spmm' with file: " << filepath << "\n";
		Input input = read_input(filepath);

		float*             a_ptr = csr_to_row_major(input.weights[0]);
		std::vector<float> a(a_ptr, a_ptr + MAT_SIZE * MAT_SIZE);
		std::free(a_ptr);
		float*             b_ptr = input.embeddings;
		std::vector<float> b(b_ptr, b_ptr + MAT_SIZE * MAT_SIZE);

		std::vector<float> expected = host_spmm(a, b, MAT_SIZE, MAT_SIZE);
		run(input);
		std::vector<float> actual(reinterpret_cast<float*>(input.data), reinterpret_cast<float*>(input.data) + MAT_SIZE * MAT_SIZE);
		ASSERT_EQ(expected, actual, "The matrices differ in values.\n");

		printf("Test successful\n");
		cuda_dealloc_host(input.data);
	}
}

int main()
{
	auto path = std::filesystem::current_path() / DATA_DIRECTORY / "dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_self_attention_multihead_attention_q.smtx";
	test_dev_spmm(path);

	return 0;
}
