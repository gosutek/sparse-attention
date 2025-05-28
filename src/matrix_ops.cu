
#include "common.h"

#include "matrix_ops.cuh"

// __float2half(const float a)
// __float2half_rd(const float a) ~ round down
// __float2half_rn(const float a) ~ round nearest
// __float2half_ru(const float a) ~ round up
// __float2half_rz(const float a) ~ round towards zero
//
// float2float

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

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

// static bool block_brick_sort(const COOElement& a, const COOElement& b)
// {
// 	const size_t row_panel_idx_a = a.row / ROW_PANEL_SIZE;
// 	const size_t row_panel_idx_b = b.row / ROW_PANEL_SIZE;
//
// 	const size_t block_row_a = a.row / TM;
// 	const size_t block_row_b = b.row / TM;
//
// 	const size_t block_col_a = a.col / TK;
// 	const size_t block_col_b = b.col / TK;
//
// 	const size_t brick_row_a = a.row / brick_m;
// 	const size_t brick_row_b = b.row / brick_m;
//
// 	const size_t brick_col_a = a.col / brick_k;
// 	const size_t brick_col_b = b.col / brick_k;
//
// 	return std::tie(
// 			   row_panel_idx_a,
// 			   block_row_a,
// 			   block_col_a,
// 			   brick_col_a,
// 			   brick_row_a,
// 			   a.col,
// 			   a.row) < std::tie(row_panel_idx_b,
// 							block_row_b,
// 							block_col_b,
// 							brick_col_b,
// 							brick_row_b,
// 							b.col,
// 							b.row);
// }
//
// static bool row_panel_sort(const COOElement& a, const COOElement& b)
// {
// 	const size_t row_panel_idx_a = a.row / ROW_PANEL_SIZE;
// 	const size_t row_panel_idx_b = b.row / ROW_PANEL_SIZE;
// 	return std::tie(row_panel_idx_a, a.col, a.row) < std::tie(row_panel_idx_b, b.col, b.row);
// }
//
// static void initialize_new_block(std::shared_ptr<HRPB> hrpb_ptr, ProcessingState& state)
// {
// 	Block& block_ref = hrpb_ptr->packed_blocks.emplace_back();
// 	block_ref.rows.reserve((TM / brick_m) * (TK / brick_k));  // Reserve the maximum amount possible for this block, i.e. the max number of bricks in a block
//
// 	state.block_idx++;    // Block indices are relative to the array, i.e. they never zero out
// 	state.brick_idx = 0;  // Brick indices are relative to the block, so this should zero out
//
// 	// brick_col_ptr
// 	state.brick_col_ptr_idx = 0;
//
// 	// block_row_ptr
// 	state.block_row_ptr_count++;  // we have entered a new block
// }
//
// static void finalize_block(std::shared_ptr<HRPB> hrpb_ptr, ProcessingState& state)
// {
// 	Block& block = hrpb_ptr->packed_blocks[static_cast<size_t>(state.block_idx)];
// 	if (state.brick_col_ptr_idx == 0)
// 		state.brick_col_ptr_idx++;
// 	for (size_t i = state.brick_col_ptr_idx; i < block.col_ptr.size(); ++i) {  // any leftovers at brick_col_ptr_idx (where we left off) should be equal to the number of bricks
// 		block.col_ptr[i] = state.brick_idx;                                    // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
// 	}
// 	hrpb_ptr->block_row_ptr.back() = hrpb_ptr->packed_blocks.size();
// 	hrpb_ptr->size_ptr.push_back(block.get_block_size() + hrpb_ptr->size_ptr.back());  // This blocks starting address is that previous block's starting address plus its size in bytes
// }

// TODO: Merge this and above
// static void finalize_last_block(std::shared_ptr<HRPB> hrpb_ptr, ProcessingState& state)
// {
// 	Block& block = hrpb_ptr->packed_blocks[static_cast<size_t>(state.block_idx)];
// 	if (state.brick_col_ptr_idx == 0)
// 		state.brick_col_ptr_idx++;
// 	for (size_t i = state.brick_col_ptr_idx; i < block.col_ptr.size(); ++i) {  // any leftovers at brick_col_ptr_idx (where we left off) should be equal to the number of bricks
// 		block.col_ptr[i] = state.brick_idx;                                    // should assign from brick_idx up to the end of col ptr with brick_idx on the last block of hrpb_ptr
// 	}
// 	hrpb_ptr->block_row_ptr.back() = hrpb_ptr->packed_blocks.size();
// }

// TODO: Should be static
// TODO: Handle case where mtx.rows % ROW_PANEL_SIZE != 0
// std::shared_ptr<HRPB> write_hrpb(COOMatrix& mtx, const std::filesystem::path& filepath)
// {
// 	std::shared_ptr<HRPB> hrpb_ptr = std::make_shared<HRPB>();  // NOTE: maybe don't allocate this on the heap?
// 	ProcessingState       state;
//
// 	hrpb_ptr->block_row_ptr.resize((mtx.rows + ROW_PANEL_SIZE - 1) / ROW_PANEL_SIZE + 1);
// 	hrpb_ptr->block_row_ptr[0] = 0;
//
// 	std::sort(mtx.elements.begin(), mtx.elements.end(), &row_panel_sort);
//
// 	uint32_t current_panel = static_cast<uint32_t>(-1);  // WARNING: intended overflow
// 	uint32_t current_col = static_cast<uint32_t>(-1);    // WARNING: intended overflow
// 	uint32_t where_i_should_go = 0;                      // Rename this shit
//
// 	/*
//      * Iterate first by row panel then by col
//      * aggregate all columns containing at least one non-zero
//      */
// 	// TODO: Refactor this
// 	for (COOElement& e : mtx.elements) {
// 		uint32_t panel_idx = e.row / ROW_PANEL_SIZE;
// 		if (panel_idx != current_panel) {  // Entered a new row panel
// 			current_panel = panel_idx;
// 			current_col = static_cast<uint32_t>(-1);
// 			where_i_should_go = static_cast<uint32_t>(-1);
// 		}
// 		if (e.col != current_col) {  // Entered a new col in the panel
// 			current_col = e.col;
// 			where_i_should_go++;
// 		}
// 		hrpb_ptr->active_cols.push_back(e.col);  // I don't think we know the size of this at compile time
// 		if (where_i_should_go == static_cast<uint32_t>(-1)) {
// 			THROW_RUNTIME_ERROR("variable 'where_i_should_go' is negative when it shouldn't");
// 		}
// 		e.col = where_i_should_go;
// 	}
//
// 	std::sort(mtx.elements.begin(), mtx.elements.end(), &block_brick_sort);
//
// 	for (const COOElement& e : mtx.elements) {
// 		const int32_t row_panel_idx = e.row / ROW_PANEL_SIZE;
// 		const int32_t block_row = e.row / TM;
// 		const int32_t block_col = e.col / TK;
// 		const int32_t brick_row = e.row / brick_m;
// 		const int32_t brick_col = e.col / brick_k;
//
// 		// Entered new row panel
// 		// since ROW_PANEL_SIZE multiple of TM we have also
// 		// entered a new block
//
// 		if (row_panel_idx != state.current_row_panel) {
// 			hrpb_ptr->block_row_ptr[row_panel_idx] = hrpb_ptr->packed_blocks.size();
// 			state.current_row_panel = row_panel_idx;
// 		}
//
// 		if (block_row != state.current_block_row || block_col != state.current_block_col) {
// 			// Block transitions
// 			if (state.block_idx > -1) {
// 				finalize_block(hrpb_ptr, state);
// 			} else [[unlikely]] {
// 				hrpb_ptr->size_ptr.push_back(0);  // the first block starts at 0 offset
// 			}
//
// 			state.current_block_row = block_row;
// 			state.current_block_col = block_col;
// 			initialize_new_block(hrpb_ptr, state);
// 		}
//
// 		if (brick_row != state.current_brick_row || brick_col != state.current_brick_col) {
// 			// Brick transitions
// 			const int32_t rel_brick_row = brick_row - (state.current_block_row * (TM / brick_m));
// 			const int32_t rel_brick_col = brick_col - (state.current_block_col * (TK / brick_k));
//
// 			if (rel_brick_col < 0 || rel_brick_row < 0)
// 				THROW_RUNTIME_ERROR("Relative row or block came back negative");
//
// 			hrpb_ptr->packed_blocks[state.block_idx].rows.push_back(rel_brick_row);
//
// 			if (brick_col != state.current_brick_col) {  // brick col changed
// 				hrpb_ptr->packed_blocks[state.block_idx].col_ptr[state.brick_col_ptr_idx] = state.brick_idx;
// 				state.brick_col_ptr_idx++;
// 			}
//
// 			state.current_brick_row = brick_row;
// 			state.current_brick_col = brick_col;
// 			state.brick_idx++;
// 		}
//
// 		const int32_t rel_elem_row = e.row % brick_m;
// 		const int32_t rel_elem_col = e.col % brick_k;
// 		const int32_t pattern_idx = rel_elem_row * brick_k + rel_elem_col;
// 		hrpb_ptr->packed_blocks[state.block_idx].nnz_array.push_back(e.val);
//
// 		hrpb_ptr->packed_blocks[state.block_idx].patterns[state.brick_idx - 1] |= (1ull << pattern_idx);
// 	}
//
// 	finalize_last_block(hrpb_ptr, state);  // final block
//
// 	return hrpb_ptr;
// 	// delete hrpb_ptr;
// }

/*
 * Converts mtx from COO to CSR format
 * Writes to filename.csr binary
 */
// TODO: Figure out how to make static
// void write_csr(COOMatrix& mtx, const std::filesystem::path& filepath)
// {
// 	std::vector<int>      row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
// 	std::vector<uint32_t> col_idx(static_cast<size_t>(mtx.nnz));
// 	// TODO: template the val?
// 	std::vector<float> val(static_cast<size_t>(mtx.nnz));
//
// 	std::sort(mtx.elements.begin(), mtx.elements.end(), [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });
//
// 	std::cout << "Populating row_ptr, col_idx, val..." << std::flush;
//
// 	for (size_t i = 0; i < mtx.elements.size(); ++i) {
// 		const auto& e = mtx.elements[i];
// 		row_ptr[static_cast<size_t>(e.row) + 1]++;
// 		col_idx[i] = e.col;
// 		val[i] = __float2half_rn(e.val);
// 	}
// 	std::cout << "Done!\n";
// 	std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.data());
//
// 	return;
// }

/*
* 1. Host reads binary into pinned memory
* 2. Deserialize
* 2. Data gets loaded into global memory
* 4. Convert to __half
*/

static size_t calculate_padding(size_t size)
{
	size_t remainder = size % ALIGNMENT;
	if (ALIGNMENT == 0 || remainder == 0) {
		return 0;
	} else {
		return ALIGNMENT - remainder;
	}
}

__global__ void deserialization_kernel(float* src_sparse_ptr, float* src_dense_ptr,
	void* sparse_pitched_mem, void* dense_pitched_mem, size_t pitch,
	size_t rows, size_t cols)
{
	// SpmmDTO dto;

	const uint32_t thread_col = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t thread_row = blockIdx.y * blockDim.y + threadIdx.y;

	// WARNING: Warp divergence shouldn't happen as long as no.threads is a multiple of 32
	// I am generating neatly sized matrices. Figure out how to minimize warp divergence when matrix shape is not a multiple of 32
	if (thread_col < cols && thread_row < rows) {
		const float value = src_sparse_ptr[thread_row * cols + thread_col];

		float* const thread_local_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(sparse_pitched_mem) + thread_row * pitch) + thread_col;
		*thread_local_ptr = value;
	} else if (thread_col < 2 * cols && thread_row < 2 * rows) {
		const float value = src_dense_ptr[thread_row * cols + thread_col];

		float* const thread_local_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(dense_pitched_mem) + thread_row * pitch) + thread_col;
		*thread_local_ptr = value;
	}
}

__host__ void read_binary(const std::filesystem::path& filepath)
{
	if (!std::filesystem::exists(filepath) || !std::filesystem::is_regular_file(filepath))
		THROW_RUNTIME_ERROR("Invalid file given");
	if (filepath.extension() != ".spmm")
		THROW_RUNTIME_ERROR("Invalid file type given, expected: '.spmm'");

	void*  host_serialized_ptr = nullptr;
	size_t filesize = std::filesystem::file_size(filepath);

	printf("File %s with a filesize of %zu will be loaded into global memory\n", filepath.c_str(), filesize);
	CUDA_CHECK(cudaMallocHost(&host_serialized_ptr, filesize));

	std::ifstream ifs(filepath, std::ios::binary);
	ifs.read(reinterpret_cast<char*>(host_serialized_ptr), filesize);

	const size_t rows = *reinterpret_cast<size_t*>(host_serialized_ptr);
	const size_t cols = *(reinterpret_cast<size_t*>(host_serialized_ptr) + 1);
	void*        dev_serialized_ptr = nullptr;

	void*  pitched_mem_start = nullptr;
	void*  dense_pitched_mem = nullptr;
	void*  sparse_pitched_mem = nullptr;
	size_t pitch = 0;

	CUDA_CHECK(cudaMalloc(&dev_serialized_ptr, filesize));  // WARNING: This wastes space if we then copy the sparse+dense elements into pitched memory

	// Access with T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column
	CUDA_CHECK(cudaMallocPitch(&pitched_mem_start, &pitch, cols * sizeof(float), 2 * rows));
	sparse_pitched_mem = pitched_mem_start;
	dense_pitched_mem = reinterpret_cast<void*>(
		(reinterpret_cast<char*>(sparse_pitched_mem) + rows * pitch));

	CUDA_CHECK(cudaMemcpy(dev_serialized_ptr, host_serialized_ptr, filesize, cudaMemcpyHostToDevice));

	dim3           block_size(16, 16);                                               // 256 threads per block
	const uint32_t blocks_to_cover_cols = (cols + block_size.x - 1) / block_size.x;  // ceil-ed
	const uint32_t blocks_to_cover_rows = (rows + block_size.y - 1) / block_size.y;  // ceil-ed
	dim3           grid_size(2 * blocks_to_cover_cols, 2 * blocks_to_cover_rows);    // there are two of them

	size_t padding = 0;
	size_t chunk_size = 0;

	chunk_size = sizeof(rows) + sizeof(cols);  // the bytes occupied by the first two members (rows and cols)
	padding = calculate_padding(chunk_size);   // padding applied to the first two members (rows and cols)

	float* src_sparse_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(dev_serialized_ptr) + chunk_size + padding);  // skip rows and cols, should point to sparse_elements

	chunk_size = rows * cols * sizeof(float);  // the size of sparse_elements in bytes
	padding = calculate_padding(chunk_size);   // padding applied to sparse_elements

	float* src_dense_ptr = src_sparse_ptr + chunk_size + padding;  // should point to dense_elements

	deserialization_kernel<<<grid_size, block_size>>>(src_sparse_ptr, src_dense_ptr, sparse_pitched_mem, dense_pitched_mem, pitch, rows, cols);
	cudaDeviceSynchronize();

	// Gimme (0,0) of sparse_elements
	float              h_test_value;
	constexpr uint8_t  t_row = 0;
	constexpr uint8_t  t_col = 0;
	const float* const test_value = reinterpret_cast<float*>(reinterpret_cast<char*>(src_sparse_ptr) + t_row * pitch) + t_col;
	CUDA_CHECK(cudaMemcpy(&h_test_value, test_value, sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << h_test_value << "\n";

	// WARNING: This should only happen once every
	// matrix needed is loaded into device memory
	// since its heavy
	cudaFreeHost(host_serialized_ptr);
	cudaFree(pitched_mem_start);
}
