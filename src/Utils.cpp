// clang-format off
#include <algorithm>
#include <cstdio>
#include "../include/mmio.h"
// clang-format on
#include <filesystem>
#include <numeric>
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

static void convert_csr(const COOMatrix& mtx)
{
	int* row_ptr = (int*)malloc(((size_t)mtx.rows + 1) * sizeof(int));
	int* col_idx = (int*)malloc((size_t)mtx.nnz * sizeof(int));
	// TODO: template the val?
	float* val = (float*)malloc((size_t)mtx.nnz * sizeof(float));

	for (size_t i = 0; i < mtx.elements.size(); ++i) {
		const auto& e = mtx.elements[i];
		row_ptr[e.row + 1]++;
		col_idx[i] = e.col;
		val[i] = e.val;
	}
	std::partial_sum(row_ptr, row_ptr + (mtx.rows + 1), row_ptr);

	free(row_ptr);
	free(col_idx);
	free(val);

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
			const auto& filename = filepath.path().string();
			COOMatrix   coo_matrix = read_mtx(filename);
			convert_csr(coo_matrix);
		}
	}
}

int main()
{
	convert_all();
	return 0;
}
