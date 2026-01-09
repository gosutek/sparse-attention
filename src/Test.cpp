#include "Test.h"
#include "matrix.h"

#include <filesystem>
#include <vector>

static void enumerate_input_space()
{
	std::filesystem::path              test_base_path{ TEST_DIR };
	std::vector<std::filesystem::path> test_space = {};

	for (const auto& entry : std::filesystem::recursive_directory_iterator(test_base_path)) {
		if (entry.is_regular_file() && entry.path().extension() == ".smtx") {
		}
	}
}
