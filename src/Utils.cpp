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
#define ROW_PANEL_SIZE 32  // I think this should be the same as TM

// bogus
#define TM 32
#define TK 16
#define brick_m 16
#define brick_k 4

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

#define THROW_RUNTIME_ERROR(message) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + message)

// TODO: Review structs
struct MatrixHeader
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	size_t row_ptr_bytes;
	size_t col_idx_bytes;
	size_t val_bytes;
	size_t dense_bytes;
};

struct COOElement
{
	uint32_t row, col;

	float val;
};

struct COOMatrix
{
	uint32_t rows, cols, nnz;

	std::vector<COOElement> elements;
};

struct Block
{
	// maybe leave the arrays I only write to uninitialized?
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns{};  // cache-friendly
	std::array<uint64_t, TK / brick_k + 1>                colPtr{};
	std::vector<uint64_t>                                 rows{};       // unknown at compile time, paper is wrong
	std::vector<float>                                    nnz_array{};  // unknown at compile time

	uint32_t get_block_size()
	{
		return static_cast<uint32_t>(sizeof(Block) + nnz_array.size() * sizeof(float));
	}
};

struct HRPB
{
	std::vector<uint32_t> block_row_ptr{};
	std::vector<uint32_t> active_cols{};
	std::vector<uint32_t> size_ptr{};
	std::vector<Block>    packed_blocks{};  // What happens if this has to resize? Do all above elements get moves aswell?
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
		THROW_RUNTIME_ERROR("Failed to open file ~ " + filepath.filename().string());

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Error reading mtx banner");
	}

	if (!mm_is_sparse(matcode)) {
		fclose(f);
		THROW_RUNTIME_ERROR("Matrix is not sparse");
	}

	int rows, cols, nnz;
	if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Failed to read matrix size");
	}

	std::vector<COOElement> elements;
	elements.reserve(static_cast<size_t>(nnz));

	std::cout << "Reading COO..." << std::flush;

	for (int i = 0; i < nnz; ++i) {
		COOElement e;
		if (fscanf(f, "%u %u %f\n", &e.row, &e.col, &e.val) != 3) {
			fclose(f);
			THROW_RUNTIME_ERROR("Error reading element ~ " + std::to_string(i));
		}
		e.row--;
		e.col--;
		elements.push_back(e);
	}
	std::cout << "Done!\n";

	fclose(f);
	return { static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(nnz), std::move(elements) };
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
void write_hrpb(COOMatrix& mtx, [[maybe_unused]] const std::filesystem::path& filepath)
{
	HRPB* hrpb_ptr = new HRPB();
	hrpb_ptr->block_row_ptr.resize(mtx.rows / TM + 1);
	std::sort(mtx.elements.begin(), mtx.elements.end(), &row_panel_sort);

	uint32_t current_panel = static_cast<uint32_t>(-1);  // WARNING: intended overflow
	uint32_t current_col = static_cast<uint32_t>(-1);    // WARNING: intended overflow
	uint32_t where_i_should_go = 0;                      // Rename this shit

	/*
     * Iterate first by row panel then by col
     * aggregate all columns containing at least one non-zero
     */
	for (COOElement& e : mtx.elements) {
		uint32_t panel_idx = e.row / ROW_PANEL_SIZE;
		if (panel_idx != current_panel) {  // Entered a new row panel
			current_panel = panel_idx;
			current_col = static_cast<uint32_t>(-1);
			where_i_should_go = static_cast<uint32_t>(-1);
		}
		if (e.col != current_col) {  // Entered a new col in the panel
			current_col = e.col;
			where_i_should_go++;
		}
		hrpb_ptr->active_cols.push_back(e.col);  // I don't think we know the size of this at compile time
		if (where_i_should_go == static_cast<uint32_t>(-1)) {
			THROW_RUNTIME_ERROR("variable 'where_i_should_go' is negative when it shouldn't");
		}
		e.col = where_i_should_go;
	}

	std::sort(mtx.elements.begin(), mtx.elements.end(), &block_brick_sort);

	uint32_t row_panel_idx = static_cast<uint32_t>(-1);  // WARNING: intended overflow

	size_t block_row = static_cast<size_t>(-1);  // WARNING: intended overflow
	size_t block_col = static_cast<size_t>(-1);  // WARNING: intended overflow

	size_t block_idx = static_cast<size_t>(-1);  // WARNING: intended overflow

	uint32_t block_row_ptr_count = 0;
	uint32_t block_row_ptr_idx = 0;

	size_t brick_row = static_cast<size_t>(-1);  // WARNING: intended overflow
	size_t brick_col = static_cast<size_t>(-1);  // WARNING: intended overflow

	size_t brick_idx = 0;

	size_t brick_colPtr_count = 0;
	size_t brick_colPtr_idx = static_cast<size_t>(-1);  // WARNING: intended overflow

	// Add descriptive comments for this mess of a for loop
	for (const COOElement& e : mtx.elements) {
		// Entered new row panel
		// since ROW_PANEL_SIZE multiple of TM we have also
		// entered a new block
		if (row_panel_idx != e.row / ROW_PANEL_SIZE) {
			row_panel_idx = e.row / ROW_PANEL_SIZE;

			hrpb_ptr->block_row_ptr[block_row_ptr_idx] = block_row_ptr_count;  // Register the blocks of the previous row_panel

			block_row_ptr_idx++;
		}

		if (block_row != e.row / TM && block_col == e.col / TK)  // if we changed ONLY the block row
		{
			for (size_t i = ++brick_colPtr_idx; i < hrpb_ptr->packed_blocks[block_idx].colPtr.size(); ++i) {  // any leftovers AFTER brick_colPtr_idx (where we left off) should be equal to the number of bricks
				hrpb_ptr->packed_blocks[block_idx].colPtr[i] = brick_idx;                                     // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
			}
			block_row = e.row / TM;
			hrpb_ptr->size_ptr.push_back(hrpb_ptr->packed_blocks[block_idx].get_block_size() + hrpb_ptr->size_ptr.back());
			hrpb_ptr->packed_blocks[block_idx].colPtr[TK / brick_k] = brick_idx;                   // when changing blocks, the last element of the previous block's colPtr vector should be equal to the number of bricks in that block
			hrpb_ptr->packed_blocks.emplace_back().rows.reserve((TM / brick_m) * (TK / brick_k));  // Reserve at the maximum possible size
			block_row_ptr_count++;
			brick_idx = 0;
			block_idx++;

			brick_colPtr_idx = 0;
			brick_colPtr_count = 0;

		} else if (block_col != e.col / TK && block_row == e.row / TM) {  // if we changed ONLY the block column

			for (size_t i = ++brick_colPtr_idx; i < hrpb_ptr->packed_blocks[block_idx].colPtr.size(); ++i) {  // any leftovers AFTER brick_colPtr_idx (where we left off) should be equal to the number of bricks
				hrpb_ptr->packed_blocks[block_idx].colPtr[i] = brick_idx;                                     // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
			}

			block_col = e.col / TK;
			hrpb_ptr->size_ptr.push_back(hrpb_ptr->packed_blocks[block_idx].get_block_size() + hrpb_ptr->size_ptr.back());
			hrpb_ptr->packed_blocks[block_idx].colPtr[TK / brick_k] = brick_idx;                   // when changing blocks, the last element of the previous block's colPtr vector should be equal to the number of bricks in that block
			hrpb_ptr->packed_blocks.emplace_back().rows.reserve((TM / brick_m) * (TK / brick_k));  // Reserve at the maximum possible size
			block_row_ptr_count++;
			brick_idx = 0;
			block_idx++;

			brick_colPtr_idx = 0;
			brick_colPtr_count = 0;

		} else if (block_row != e.row / TM && block_col != e.col / TK) {  // if we change both row and column of a block. Happens when we reach the right side of the matrix AND must enter a new block

			if (block_idx != static_cast<size_t>(-1)) {
				for (size_t i = ++brick_colPtr_idx; i < hrpb_ptr->packed_blocks[block_idx].colPtr.size(); ++i) {  // any leftovers AFTER brick_colPtr_idx (where we left off) should be equal to the number of bricks
					hrpb_ptr->packed_blocks[block_idx].colPtr[i] = brick_idx;                                     // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
				}
			}

			block_row = e.row / TM;
			block_col = e.col / TK;
			if (!hrpb_ptr->size_ptr.empty()) {
				hrpb_ptr->size_ptr.push_back(hrpb_ptr->packed_blocks[block_idx].get_block_size() + hrpb_ptr->size_ptr.back());
			} else {
				hrpb_ptr->size_ptr.push_back(0);
			}
			hrpb_ptr->packed_blocks.emplace_back().rows.reserve((TM / brick_m) * (TK / brick_k));  // Reserve at the maximum possible size
			block_row_ptr_count++;
			brick_idx = 0;
			block_idx++;

			brick_colPtr_idx = 0;
			brick_colPtr_count = 0;
		}

		if (brick_row != e.row / brick_m && brick_col == e.col / brick_k) {  // Changed brick row ONLY (down)
			brick_row = e.row / brick_m;

			hrpb_ptr->packed_blocks[block_idx].rows.push_back(brick_row);
			brick_idx++;
			brick_colPtr_count++;
		} else if (brick_col != e.col / brick_k && brick_row == e.row / brick_m) {  // Changed brick column ONLY
			brick_col = e.col / brick_k;
			hrpb_ptr->packed_blocks[block_idx].rows.push_back(brick_row);
			brick_idx++;

			brick_colPtr_idx++;
			brick_colPtr_count++;
			hrpb_ptr->packed_blocks[block_idx].colPtr[brick_colPtr_idx] = brick_colPtr_count;
		} else if (brick_row != e.row / brick_m && brick_col != e.col / brick_k) {  // Changed BOTH column and row
			brick_row = e.row / brick_m;
			brick_col = e.col / brick_k;

			hrpb_ptr->packed_blocks[block_idx].rows.push_back(brick_row);
			brick_idx++;

			brick_colPtr_idx++;
			brick_colPtr_count++;
			hrpb_ptr->packed_blocks[block_idx].colPtr[brick_colPtr_idx] = brick_colPtr_count;
		}
		/*
         * 1. How do I store past blocks? In an std::vector<Block>
         * 2. Maybe write them to binary
         * while processing the next one?
         */

		hrpb_ptr->packed_blocks[block_idx].nnz_array.push_back(e.val);

		size_t e_relative_row;  // relative to brick
		size_t e_relative_col;  // relative to brick

		size_t brick_relative_row;  // relative to block, should have a range [0..1]
		size_t brick_relative_col;  // relative to block, should have a range [0..3]

		// TODO: Rearrange the if, the most common should be on the top
		if (brick_row == 0) {  // add unlikely
			e_relative_row = e.row;
		} else {
			e_relative_row = e.row - brick_row * brick_m;
		}

		if (brick_col == 0) {  // add unlikely
			e_relative_col = e.col;
		} else {
			e_relative_col = e.col - brick_col * brick_k;
		}

		if (block_row == 0) {  // add unlikely
			brick_relative_row = brick_row;
		} else {
			brick_relative_row = brick_row - block_row * (TM / brick_m);
		}

		if (block_col == 0) {  // add unlikely
			brick_relative_col = brick_col;
		} else {
			brick_relative_col = brick_col - block_col * (TK / brick_k);
		}

		size_t e_row_major_idx = e_relative_row * brick_k + e_relative_col;
		hrpb_ptr->packed_blocks[block_idx].patterns[brick_relative_row * (TK / brick_k) + brick_relative_col] |= static_cast<uint64_t>(1) << e_row_major_idx;
	}

	for (size_t i = ++brick_colPtr_idx; i < hrpb_ptr->packed_blocks[block_idx].colPtr.size(); ++i) {  // any leftovers AFTER brick_colPtr_idx (where we left off) should be equal to the number of bricks
		hrpb_ptr->packed_blocks[block_idx].colPtr[i] = brick_idx;                                     // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
	}
	// do the same for remaining block_row_ptr
	hrpb_ptr->block_row_ptr[block_row_ptr_idx] = block_row_ptr_count;
	delete hrpb_ptr;
}

/*
 * Converts mtx from COO to CSR format
 * Writes to filename.csr binary
 */
// TODO: Figure out how to make static
void write_csr(COOMatrix& mtx, const std::filesystem::path& filepath)
{
	std::vector<int>      row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
	std::vector<uint32_t> col_idx(static_cast<size_t>(mtx.nnz));
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
		THROW_RUNTIME_ERROR("Failed to open file ~ " + filepath.filename().string());

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Error reading mtx banner");
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
