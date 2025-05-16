#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_fp16.h"
#include "mmio.h"
// __float2half(const float a)
// __float2half_rd(const float a) ~ round down
// __float2half_rn(const float a) ~ round nearest
// __float2half_ru(const float a) ~ round up
// __float2half_rz(const float a) ~ round towards zero
//
// __half2float

#define LIKELY(x) __builtin_expect(!!(x), 1)

#define DATA_DIRECTORY "data/"
#define ALIGNMENT 128
#define ROW_PANEL_SIZE 1024

// bogus
#define TM 32
#define TK 16
#define brick_m 16
#define brick_k 4
#define M 1000
#define NUM_BLKS 16

/*
 * C = A*B
 * MxNxK
 * where 
 * A is MxK
 * B is KxN
 * C is MxN
 */

#ifndef BSR_BLOCK_SIZE
#	define BSR_BLOCK_SIZE 2
#endif

// TODO: Review structs
struct MatrixHeader
{
	int32_t rows;
	int32_t cols;
	int64_t nnz;

	size_t row_ptr_bytes;
	size_t col_idx_bytes;
	size_t val_bytes;
	size_t dense_bytes;
};

struct COOElement
{
	int row, col;

	float val;
};

struct COOMatrix
{
	int rows, cols, nnz;

	std::vector<COOElement> elements;
};

struct Block
{
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns;  // why is this in [][] format?
	std::array<uint, TK / brick_k + 1>                    colPtr;
	std::array<uint, (TM / brick_m) * (TK / brick_k)>     rows;
	std::vector<float>                                    nnz_array;  // unknown at compile time

	Block() :
		patterns{}, colPtr{}, rows{} {}  // Zero initialize them
										 // Might not be needed for colPtr, rows
										 // DEFINATELY needed for patterns
};

struct HRPB
{
	void* packedBlocks;
	uint  blockedRowPtr[M / TK + 1];
	uint  activeCols[NUM_BLKS * TK];
	uint  sizePtr[NUM_BLKS + 1];
};

static bool block_brick_sort(const COOElement& a, const COOElement& b)
{
	const size_t row_panel_idx_a = a.row / ROW_PANEL_SIZE;
	const size_t row_panel_idx_b = b.row / ROW_PANEL_SIZE;

	const size_t block_row_a = a.row / TM;
	const size_t block_row_b = b.row / TM;

	const size_t block_col_a = a.col / TK;
	const size_t block_col_b = b.col / TK;

	const size_t brick_row_a = a.row / brick_m;
	const size_t brick_row_b = b.row / brick_m;

	const size_t brick_col_a = a.col / brick_k;
	const size_t brick_col_b = b.col / brick_k;

	return std::tie(
			   row_panel_idx_a,
			   block_row_a,
			   block_col_a,
			   brick_col_a,
			   brick_row_a,
			   a.col,
			   a.row) < std::tie(row_panel_idx_b,
							block_row_b,
							block_col_b,
							brick_col_b,
							brick_row_b,
							b.col,
							b.row);
}

static bool row_panel_sort(const COOElement& a, const COOElement& b)
{
	const size_t row_panel_idx_a = a.row / ROW_PANEL_SIZE;
	const size_t row_panel_idx_b = b.row / ROW_PANEL_SIZE;
	return std::tie(row_panel_idx_a, a.col, a.row) < std::tie(row_panel_idx_b, b.col, b.row);
}

// TODO: Parallelize?
static void write_binary_aligned(std::ofstream& file, const void* data, size_t size, size_t alignment)
{
	file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));

	// TODO: Review arithmetic
	size_t            padding = (alignment - (size % alignment)) % alignment;
	std::vector<char> zeros(padding, 0);
	file.write(zeros.data(), static_cast<std::streamsize>(padding));
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
	elements.reserve(static_cast<size_t>(nnz));

	std::cout << "Reading COO..." << std::flush;

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
	std::cout << "Done!\n";

	fclose(f);
	return { rows, cols, nnz, std::move(elements) };
}

/*
 * Generates a dense matrix in column-major format
 * of size rows * cols filled with random values
 */
// TODO: Parallelize?
// TODO: Made static after testing
static std::vector<__half> generate_dense(size_t size)
{
	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	std::vector<__half> dense_values;
	std::cout << "Generating Dense Matrix..." << std::flush;
	dense_values.reserve(size);
	for (size_t i = 0; i < size; ++i) {
		__half half_random_value = __float2half_rn(uni_real_dist(rng));
		dense_values.push_back(half_random_value);
	}
	std::cout << "Done!" << std::endl;
	;

	return dense_values;
}

// TODO: Figure out how to make static
void write_hrpb(COOMatrix& mtx, const std::filesystem::path& filepath)
{
	std::sort(mtx.elements.begin(), mtx.elements.end(), &row_panel_sort);
	std::vector<int> active_col_idx;

	size_t row_panel_num = (mtx.rows + ROW_PANEL_SIZE - 1) / ROW_PANEL_SIZE;
	size_t current_panel = -1;
	size_t current_col = -1;
	size_t where_i_should_go = 0;  // Rename this shit

	/*
     * Iterate first by row panel then by col
     * aggregate all columns containing at least one non-zero
     */
	size_t count = 0;
	for (COOElement& e : mtx.elements) {
		size_t panel_idx = e.row / ROW_PANEL_SIZE;
		if (panel_idx != current_panel) {  // Entered a new row panel
			current_panel = panel_idx;
			current_col = -1;
			where_i_should_go = -1;
		}
		if (e.col != current_col) {  // Entered a new col in the panel
			current_col = e.col;
			where_i_should_go++;
		}
		active_col_idx.push_back(e.col);  // I don't think we know the size of this at compile time
		e.col = where_i_should_go;
	}

	/*
     * 1. Sort again after row panelling
     * 2. Iterate normally
     * 3. For each element calculate IN WHICH BLOCK AND BRICK THEY BELONG TO
     * 4. Fill pattern array
     */

	std::sort(mtx.elements.begin(), mtx.elements.end(), &block_brick_sort);

	size_t block_row = 0;
	size_t block_col = 0;

	size_t brick_row = 0;
	size_t brick_col = 0;

	size_t brick_idx = 0;

	size_t brick_colPtr_count = 0;  // we handle the first brick out of the loop
	size_t brick_colPtr_idx = 0;    // we handle the first brick out of the loop

	Block* block_ptr = new Block();
	// Add the brick data of the first element
	block_ptr->rows[brick_idx] = mtx.elements[0].row / brick_m;
	brick_idx++;
	brick_colPtr_count++;

	for (const COOElement& e : mtx.elements) {
		if (block_row < e.row / TM)  // Moved down one block
		{
			block_row = e.row / TM;
			// Should create a new block here
			break;                          // let's do for one block only
		} else if (block_col < e.col / TK)  // Moved right one block
		{
			block_col = e.col / TK;
			// Should create a new block here
			break;  // let's do for one block only
		}

		if (brick_row < e.row / brick_m) {  // Moved down one brick
			brick_row = e.row / brick_m;

			block_ptr->rows[brick_idx] = brick_row;
			brick_idx++;
			brick_colPtr_count++;
			printf("Moved down for (%d, %d)\n", e.row, e.col);
		} else if (brick_col < e.col / brick_k) {  // Moved right one brick
			brick_col = e.col / brick_k;

			block_ptr->rows[brick_idx] = brick_row;
			brick_idx++;
			brick_colPtr_idx++;

			block_ptr->colPtr[brick_colPtr_idx] = brick_colPtr_count;
			brick_colPtr_count++;
		}

		block_ptr->nnz_array.push_back(e.val);

		size_t e_relative_row;
		size_t e_relative_col;
		if (brick_row == 0) {  // add unlikely
			e_relative_row = e.row;
		} else {
			e_relative_row = brick_row * brick_m - e.row;
		}

		if (brick_col == 0) {  // add unlikely
			e_relative_col = e.col;
		} else {
			e_relative_col = brick_col * brick_k - e.col;
		}
		size_t e_row_major_idx = e_relative_row * brick_k + e_relative_col;
		block_ptr->patterns[brick_row * (TM / brick_m) + brick_col] |= 1 << e_row_major_idx;
	}
	delete block_ptr;
}

/*
 * Converts mtx from COO to CSR format
 * Writes to filename.csr binary
 */
// TODO: Figure out how to make static
void write_csr(COOMatrix& mtx, const std::filesystem::path& filepath)
{
	std::vector<int> row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
	std::vector<int> col_idx(static_cast<size_t>(mtx.nnz));
	// TODO: template the val?
	std::vector<__half> val(static_cast<size_t>(mtx.nnz));

	std::sort(mtx.elements.begin(), mtx.elements.end(), [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });

	std::cout << "Populating row_ptr, col_idx, val..." << std::flush;

	for (size_t i = 0; i < mtx.elements.size(); ++i) {
		const auto& e = mtx.elements[i];
		row_ptr[static_cast<size_t>(e.row) + 1]++;
		col_idx[i] = e.col;
		val[i] = __float2half_rn(e.val);
	}
	std::cout << "Done!\n";
	std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.data());

	std::vector<__half> dense = generate_dense(static_cast<size_t>(mtx.rows * mtx.cols));

	// NOTE: trunc flag should be redundant
	std::ofstream file(filepath.parent_path() / filepath.filename().replace_extension(".csr"), std::ios::binary | std::ios::trunc);

	MatrixHeader header = {
		mtx.rows,
		mtx.cols,
		mtx.nnz,
		row_ptr.size() * sizeof(int),
		col_idx.size() * sizeof(int),
		val.size() * sizeof(int),
		(static_cast<size_t>(mtx.rows * mtx.cols)) * sizeof(__half)
	};

	write_binary_aligned(file, &header, sizeof(header), ALIGNMENT);
	write_binary_aligned(file, row_ptr.data(), header.row_ptr_bytes, ALIGNMENT);
	write_binary_aligned(file, col_idx.data(), header.col_idx_bytes, ALIGNMENT);
	write_binary_aligned(file, val.data(), header.val_bytes, ALIGNMENT);
	write_binary_aligned(file, dense.data(), header.dense_bytes, ALIGNMENT);

	file.close();

	return;
}

/*
 * Checks if [path] has a .mtx extension 
 * and
 * if the same filename exists as a .csr
 */
static bool requires_conversion(const std::filesystem::path& path)
{
	// TODO: Add a check for .bsr when it's implemented
	// Add a check for KBSR when it's implemented
	return path.extension().string() == ".mtx" &&
	       !(std::filesystem::exists(path.parent_path() / path.filename().replace_extension(".csr")));
}

/*
 * Will iterate over all data/ *.mtx matrices
 * and convert them to .bcsr format
 */
void convert(const std::filesystem::directory_iterator& target_dir, void (*conversion_func_ptr)(COOMatrix& mtx, const std::filesystem::path& filepath))
{
	for (const auto& filepath : std::filesystem::directory_iterator(target_dir)) {
		if (filepath.is_regular_file() && requires_conversion(filepath.path())) {
			COOMatrix coo_matrix = read_mtx(filepath.path());
			conversion_func_ptr(coo_matrix, filepath.path());
		}
	}
}

void print_matrix_specs(const std::filesystem::path& filepath)
{
	FILE* f = fopen(filepath.c_str(), "r");
	int   rows = 0;
	int   cols = 0;
	int   nnz = 0;

	if (!f)
		throw std::runtime_error("Failed to open file " + filepath.filename().string());

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		throw std::runtime_error("Error reading mtx banner");
	}

	std::cout << filepath.filename() << "\n";
	for (int i = 0; i < 4; ++i) {
		std::cout << matcode[i];
	}
	std::cout << "\n";

	if (mm_is_sparse(matcode)) {
		mm_read_mtx_crd_size(f, &rows, &cols, &nnz);
		std::cout << "Sparse with " << rows << " rows and " << cols << " cols and nnz " << nnz << "\n";
		std::cout << "Data type " << matcode[2] << "\n";
	} else {
		mm_read_mtx_array_size(f, &rows, &cols);
		std::cout << "Dense with " << rows << " rows and " << cols << " cols.\n";
	}
}
