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
	int row, col;

	float val;
};

struct COOMatrix
{
	int rows = 0;
	int cols = 0;
	int nnz = 0;

	std::vector<COOElement> elements;
};

static COOMatrix read_mtx(const std::string& filename)
{
	FILE* f = fopen(filename.c_str(), "r");
	if (!f)
		throw std::runtime_error("Failed to open file " + filename);

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		throw std::runtime_error("Error reading mtx banner");
	}

	if (!mm_is_sparse(matcode)) {
		fclose(f);
		throw std::runtime_error("Matrix is not sparse");
	}

	int rows, cols, nnz;
	if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
		fclose(f);
		throw std::runtime_error("Failed to read matrix size");
	}

	std::vector<COOElement> elements;
	elements.reserve(nnz);

	for (int i = 0; i < nnz; ++i) {
		COOElement e;
		if (fscanf(f, "%d %d %f\n", &e.row, &e.col, &e.val) != 3) {
			fclose(f);
			throw std::runtime_error("Error reading element " + std::to_string(i));
		}
		e.row--;
		e.col--;
		elements.push_back(e);
	}

	fclose(f);
	return { rows, cols, nnz, std::move(elements) };
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
