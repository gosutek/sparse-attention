#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "./include/json.hpp"

#include "../src/Utils.h"

#define THROW_RUNTIME_ERROR(message) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + message)

#define ASSERT_EQ(expected, actual, message)                           \
	do {                                                               \
		if (!((expected) == (actual))) {                               \
			std::cerr << "Assertion failed: " << message << std::endl; \
			std::cerr << " Expected: " << expected << std::endl;       \
			std::cerr << " Actual:   " << actual << std::endl;         \
			exit(EXIT_FAILURE);                                        \
		}                                                              \
	} while (0)

struct ReadExpectedOutput
{
	int32_t no_blocks = -1;
	HRPB    hrpb;
};

static ReadExpectedOutput read_expected(const std::filesystem::path& expected_filepath)
{
	std::ifstream  input_stream(expected_filepath, std::ios::in);
	nlohmann::json j;
	input_stream >> j;

	const int32_t no_blocks = j["no_blocks"].get<int32_t>();

	HRPB hrpb;
	hrpb.block_row_ptr = j["block_row_ptr"].get<std::vector<uint32_t>>();
	hrpb.active_cols = j["active_cols"].get<std::vector<uint32_t>>();
	hrpb.size_ptr = j["size_ptr"].get<std::vector<uint32_t>>();

	for (int32_t i = 0; i < no_blocks; ++i) {
		Block          block;
		nlohmann::json i_block = j["blocks"][static_cast<size_t>(i)];
		block.patterns = i_block["patterns"].get<std::array<uint64_t, 8>>();
		block.nnz_array = i_block["nnz_array"].get<std::vector<float>>();
		block.col_ptr = i_block["col_ptr"].get<std::array<uint64_t, 5>>();
		block.rows = i_block["rows"].get<std::vector<uint64_t>>();
		hrpb.packed_blocks.push_back(block);
	}

	return { no_blocks, hrpb };
}

static void unit_test_hrpb(const std::filesystem::directory_iterator& test_filepath)
{
	for (const auto& file : test_filepath) {
		if (file.is_regular_file() && file.path().extension() == ".mtx") {
			const auto expected_file = file.path().parent_path() / file.path().filename().replace_extension(".json");
			if (!std::filesystem::exists(expected_file)) {
				continue;
				THROW_RUNTIME_ERROR("Unit test not found for file" + expected_file.string());
			}

			std::cout << "Testing for file: " << file.path() << "\n";
			COOMatrix          mtx = read_mtx(file.path());
			HRPB*              hrpb_ptr = write_hrpb(mtx, file.path());
			ReadExpectedOutput expected = read_expected(expected_file);

			ASSERT_EQ(expected.no_blocks, hrpb_ptr->packed_blocks.size(), "Block size mismatch");
			ASSERT_EQ(expected.hrpb, *hrpb_ptr, "Structs are different");

			printf("Test successful\n");

			delete hrpb_ptr;
		}
	}
}

int main()
{
	const auto testing_path = std::filesystem::current_path() / "test/test_cases/";
	unit_test_hrpb(std::filesystem::directory_iterator(testing_path));
	return 0;
}
