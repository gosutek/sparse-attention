#include <algorithm>
// clang-format off
#include <iostream>
#include "../include/mmio.h"
// clang-format on
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define DATA_DIRECTORY "data/"
#define ALIGNMENT 128

#ifndef BSR_BLOCK_SIZE
#	define BSR_BLOCK_SIZE 2
#endif

struct MatrixHeader
{
	int32_t rows;
	int32_t cols;
	int64_t nnz;
	size_t  row_ptr_bytes;
	size_t  col_idx_bytes;
	size_t  val_bytes;
};

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

static void write_binary_aligned(std::ofstream& file, const void* data, size_t size, size_t alignment)
{
	file.write(reinterpret_cast<const char*>(data), (std::streamsize)size);

	// TODO: Review arithmetic
	size_t            padding = (alignment - (size % alignment)) % alignment;
	std::vector<char> zeros(padding, 0);
	file.write(zeros.data(), (std::streamsize)padding);
}

static COOMatrix read_mtx(const std::filesystem::path& filepath)
{
	FILE* f = fopen(filepath.c_str(), "r");
	if (!f)
		throw std::runtime_error("Failed to open file " + filepath.filename().string());

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
	elements.reserve((size_t)nnz);

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
	std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });
	return { rows, cols, nnz, std::move(elements) };
}

/*
 * Converts mtx from COO to CSR format
 * Writes to filename.csr binary
 */
static void write_csr(const COOMatrix& mtx, const std::filesystem::path& filepath)
{
	std::vector<int> row_ptr((size_t)mtx.rows + 1, 0);
	std::vector<int> col_idx((size_t)mtx.nnz);
	// TODO: template the val?
	std::vector<float> val((size_t)mtx.nnz);

	for (size_t i = 0; i < mtx.elements.size(); ++i) {
		const auto& e = mtx.elements[i];
		row_ptr[(size_t)e.row + 1]++;
		col_idx[i] = e.col;
		val[i] = e.val;
	}
	std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.data());

	std::ofstream file(DATA_DIRECTORY + filepath.filename().replace_extension(".csr").string(), std::ios::binary | std::ios::trunc);

	MatrixHeader header = {
		mtx.rows,
		mtx.cols,
		mtx.nnz,
		row_ptr.size() * sizeof(int),
		col_idx.size() * sizeof(int),
		val.size() * sizeof(int)
	};
	file.write(reinterpret_cast<const char*>(&header), sizeof(header));

	write_binary_aligned(file, row_ptr.data(), header.row_ptr_bytes, ALIGNMENT);
	write_binary_aligned(file, col_idx.data(), header.col_idx_bytes, ALIGNMENT);
	write_binary_aligned(file, val.data(), header.val_bytes, ALIGNMENT);

	file.close();

	return;
}

/*
 * Will iterate over all data/ *.mtx matrices
 * and convert them to .bcsr format
 */
static void convert_all()
{
	std::vector<std::string> file_paths;

	auto project_dir = std::filesystem::current_path();
	auto target_dir = project_dir / DATA_DIRECTORY;

	for (const auto& filepath : std::filesystem::directory_iterator(target_dir)) {
		if (filepath.is_regular_file() && filepath.path().extension().string() == ".mtx") {
			COOMatrix coo_matrix = read_mtx(filepath.path());
			write_csr(coo_matrix, filepath.path());
		}
	}
}

int main()
{
	convert_all();
	return 0;
}
