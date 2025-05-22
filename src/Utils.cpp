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

struct ProcessingState
{
	int32_t current_row_panel = -1;

	int32_t current_block_row = -1;
	int32_t current_block_col = -1;

	int32_t current_brick_row = -1;
	int32_t current_brick_col = -1;

	int32_t block_idx = -1;
	size_t  brick_idx = 0;

	size_t brick_col_ptr_idx = 0;

	size_t block_row_ptr_idx = 0;
	size_t block_row_ptr_count = 0;
};

struct Block
{
	// maybe leave the arrays I only write to uninitialized?
	std::array<uint64_t, (TM / brick_m) * (TK / brick_k)> patterns{};  // cache-friendly
	std::array<uint64_t, TK / brick_k + 1>                col_ptr{};
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

static void initialize_new_block(HRPB* hrpb_ptr, ProcessingState& state)
{
	Block& block_ref = hrpb_ptr->packed_blocks.emplace_back();
	block_ref.rows.reserve((TM / brick_m) * (TK / brick_k));  // Reserve the maximum amount possible for this block, i.e. the max number of bricks in a block

	state.block_idx++;    // Block indices are relative to the array, i.e. they never zero out
	state.brick_idx = 0;  // Brick indices are relative to the block, so this should zero out

	// brick_col_ptr
	state.brick_col_ptr_idx = 0;

	// block_row_ptr
	state.block_row_ptr_count++;  // we have entered a new block
}

static void finalize_block(HRPB* hrpb_ptr, ProcessingState& state)
{
	Block& block = hrpb_ptr->packed_blocks[state.block_idx];
	for (size_t i = state.brick_col_ptr_idx; i < block.col_ptr.size(); ++i) {  // any leftovers at brick_col_ptr_idx (where we left off) should be equal to the number of bricks
		block.col_ptr[i] = state.brick_idx;                                    // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
	}
	hrpb_ptr->size_ptr.push_back(block.get_block_size() + hrpb_ptr->size_ptr.back());  // This blocks starting address is that previous block's starting address plus its size in bytes
}

// TODO: Figure out how to make static
// TODO: Handle case where mtx.rows % ROW_PANEL_SIZE != 0
// TODO: Write unit tests
void write_hrpb(COOMatrix& mtx, [[maybe_unused]] const std::filesystem::path& filepath)
{
	HRPB*           hrpb_ptr = new HRPB();
	ProcessingState state;

	hrpb_ptr->block_row_ptr.resize((mtx.rows + ROW_PANEL_SIZE - 1) / ROW_PANEL_SIZE + 1);
	hrpb_ptr->block_row_ptr[0] = 0;

	std::sort(mtx.elements.begin(), mtx.elements.end(), &row_panel_sort);

	uint32_t current_panel = static_cast<uint32_t>(-1);  // WARNING: intended overflow
	uint32_t current_col = static_cast<uint32_t>(-1);    // WARNING: intended overflow
	uint32_t where_i_should_go = 0;                      // Rename this shit

	/*
     * Iterate first by row panel then by col
     * aggregate all columns containing at least one non-zero
     */
	// TODO: Refactor this
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

	for (const COOElement& e : mtx.elements) {
		const int32_t row_panel_idx = e.row / ROW_PANEL_SIZE;
		const int32_t block_row = e.row / TM;
		const int32_t block_col = e.col / TK;
		const int32_t brick_row = e.row / brick_m;
		const int32_t brick_col = e.col / brick_k;

		// Entered new row panel
		// since ROW_PANEL_SIZE multiple of TM we have also
		// entered a new block

		if (row_panel_idx != state.current_row_panel) {
			hrpb_ptr->block_row_ptr[row_panel_idx] = hrpb_ptr->packed_blocks.size();
			state.current_row_panel = row_panel_idx;
		}

		if (block_row != state.current_block_row || block_col != state.current_block_col) {
			// Block transitions
			if (state.block_idx > -1) {
				finalize_block(hrpb_ptr, state);
			} else {
				hrpb_ptr->size_ptr.push_back(0);  // the first block starts at 0 offset
			}

			state.current_block_row = block_row;
			state.current_block_col = block_col;
			initialize_new_block(hrpb_ptr, state);
		}

		if (brick_row != state.current_brick_row || brick_col != state.current_brick_col) {
			// Brick transitions
			const int32_t rel_brick_row = brick_row - (state.current_block_row * (TM / brick_m));
			const int32_t rel_brick_col = brick_col - (state.current_block_col * (TK / brick_k));

			if (rel_brick_col < 0 || rel_brick_row < 0)
				THROW_RUNTIME_ERROR("Relative row or block came back negative");

			hrpb_ptr->packed_blocks[state.block_idx].rows.push_back(rel_brick_row);

			if (brick_col != state.current_brick_col) {  // brick col changed
				hrpb_ptr->packed_blocks[state.block_idx].col_ptr[state.brick_col_ptr_idx] = state.brick_idx;
				state.brick_col_ptr_idx++;
			}

			state.current_brick_row = brick_row;
			state.current_brick_col = brick_col;
			state.brick_idx++;
		}

		const int32_t rel_elem_row = e.row % brick_m;
		const int32_t rel_elem_col = e.col % brick_k;
		const int32_t pattern_idx = rel_elem_row * brick_k + rel_elem_col;
		hrpb_ptr->packed_blocks[state.block_idx].nnz_array.push_back(e.val);

		hrpb_ptr->packed_blocks[state.block_idx].patterns[state.brick_idx - 1] |= (1ull << pattern_idx);
	}

	finalize_block(hrpb_ptr, state);  // final block
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
