// clang-format off
#include <cstdio>
#include "../include/mmio.h"
// clang-format on
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define DATA_DIRECTORY "data/"

#ifndef BSR_BLOCK_SIZE
#	define BSR_BLOCK_SIZE 2
#endif

struct COOElement
{
	uint32_t row = 0;
	uint32_t col = 0;

	float val = 0.f;
};

static std::vector<COOElement> read_mtx(const std::string& filename, int rows, int cols, int nnz)
{
	FILE* f;
	f = fopen(filename.c_str(), "r");

	if (f == NULL) {
		throw std::runtime_error("Failed to open file");
	}

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		throw std::runtime_error("Matrix not sparse");
	}

	if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
		throw std::runtime_error("Failed to read matrix size");
	}
	fclose(f);
	std::vector<COOElement> coo_vec;
	coo_vec.reserve(nnz);
	for (int i = 0; i < nnz; ++i) {
		COOElement coo_e;
		fscanf(f, "%u %u %g\n", &coo_e.row, &coo_e.col, &coo_e.val);
		coo_e.row--;
		coo_e.col--;
		coo_vec.push_back(coo_e);
	}
	return coo_vec;
}

static void convert_and_write_bcsr(const std::string& filename)
{
	int rows = 0;
	int cols = 0;
	int nnz = 0;

	if (rows % BSR_BLOCK_SIZE != 0) {  // if the blocks don't fit perfectly
									   // pad accordingly
	}

	if (cols % BSR_BLOCK_SIZE != 0) {
		// pad accordingly
	}

	// Convert here
	// Write to binary
	std::ofstream out(filename, std::ios::binary);
	out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
	out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
	out.write(reinterpret_cast<const char*>(&block_rows), sizeof(int32_t));
	out.write(reinterpret_cast<const char*>(&block_cols), sizeof(int32_t));
}

/*
 * Will iterate over all data/ *.mtx matrices
 * and convert them to .bcsr format
 */
static void convert_all()
{
	std::vector<std::string> file_paths;

	auto project_dir = std::filesystem::current_path().parent_path();
	auto target_dir = project_dir / DATA_DIRECTORY;

	for (const auto& filepath : std::filesystem::directory_iterator(target_dir)) {
		if (filepath.is_regular_file() && filepath.path().extension().string() == ".mtx") {
			convert_and_write_bcsr(filepath.path().string());
			std::cout << filepath.path().string() << "\n";
		}
	}
}

int main()
{
	convert_all();
	return 0;
}
