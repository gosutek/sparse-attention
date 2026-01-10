#include "Test.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

void enumerate_input_space()
{
	std::cout << std::format("Base Test Directory: {}\n", TEST_DIR);
	std::filesystem::path              test_base_path{ TEST_DIR };
	std::vector<std::filesystem::path> test_space = {};

	for (const auto& entry : std::filesystem::recursive_directory_iterator(test_base_path)) {
		if (entry.is_regular_file() && entry.path().extension() == ".smtx") {
		}
	}
}
