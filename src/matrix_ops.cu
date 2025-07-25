
#include "matrix_ops.cuh"
#include <algorithm>
#include <numeric>
#include <random>

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

struct LoadBinaryOutput
{
	void*  global_ptr = nullptr;
	size_t rows{};
	size_t cols{};
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

/*
 * Converts mtx from COO to CSR format
 * TODO: Figure out how to make static
 * TODO: Actually return something
 */
void coo_to_csr(COOMatrix& mtx)
{
	std::vector<int>      row_ptr(static_cast<size_t>(mtx.rows) + 1, 0);
	std::vector<uint32_t> col_idx(static_cast<size_t>(mtx.nnz));

	std::vector<float> val(static_cast<size_t>(mtx.nnz));

	std::sort(mtx.elements.begin(), mtx.elements.end(), [](const auto& a, const auto& b) { return std::tie(a.row, a.col) < std::tie(b.row, b.col); });

	for (size_t i = 0; i < mtx.elements.size(); ++i) {
		const auto& e = mtx.elements[i];
		row_ptr[static_cast<size_t>(e.row) + 1]++;
		col_idx[i] = e.col;
		val[i] = e.val;
	}
	std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.data());

	return;
}

CSRMatrix dlmc_to_csr(const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		THROW_RUNTIME_ERROR("Error opening file.\n");
	}

	CSRMatrix csr_matrix;

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	std::string header_line{}, line{}, token{};

	std::getline(file_stream, header_line);
	std::istringstream header_stream(header_line);
	std::getline(header_stream, token, ',');
	csr_matrix.cols = std::stoi(token);
	std::getline(header_stream, token, ',');
	csr_matrix.rows = std::stoi(token);
	std::getline(header_stream, token, ',');
	csr_matrix.nnz = std::stoi(token);

	csr_matrix.row_ptr.reserve(csr_matrix.rows + 1);
	csr_matrix.col_idx.reserve(csr_matrix.nnz);
	csr_matrix.val.reserve(csr_matrix.nnz);

	std::getline(file_stream, line);
	std::istringstream row_ptr_line(line);

	while (row_ptr_line >> token) {
		csr_matrix.row_ptr.push_back(std::stoi(token));
	}

	std::getline(file_stream, line);
	std::istringstream col_idx_line(line);

	while (col_idx_line >> token) {
		csr_matrix.col_idx.push_back(std::stoi(token));
		csr_matrix.val.push_back(uni_real_dist(rng));
	}

	return csr_matrix;
}

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
//
// // TODO: Merge this and above
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

/*
 * Say I have "cols" columns to cover. Split cols to 2D thread blocks where each column of the thread block is assigned blockDim.y elements of the matrix column.
 * Basically split the matrix into 2D threadblocks.
 * Only when a thread happens upon a non-zero value do we write to the predicate, that way no hazards happen.
 * Once the predicate matrix is calculated we split it and assign it to 1D thread blocks. Each thread block calculates the exclusive sum independently of others.
 * To fix the offset problem, when we move the columns to their intended places (packed on the left) we increment the column indicated by predicate[] like this
 * col_to_end_up_to = predicate[i] + blockIdx.x * blockDim.x
 */
static __global__ void predicate_kernel(PitchedMatrix* pcm_sparse, bool* predicate)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= pcm_sparse->cols)
		return;

	__half* col_ptr = reinterpret_cast<__half*>(reinterpret_cast<char*>(pcm_sparse->data) + col * pcm_sparse->pitch);
	if (col_ptr[row] != __float2half(0.0f)) {
		predicate[col] = true;
	}
}

// // TODO: Should be static
// // TODO: Handle case where mtx.rows % ROW_PANEL_SIZE != 0
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
// 	uint32_t current_panel = static_cast<uint32_t>(-1);  // NOTE: intended overflow
// 	uint32_t current_col = static_cast<uint32_t>(-1);    // NOTE: intended overflow
// 	uint32_t where_i_should_go = 0;                      // TODO: Rename this shit
//
// 	/*
//       * Iterate first by row panel then by col
//       * aggregate all columns containing at least one non-zero
//       */
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

static size_t calculate_padding(size_t size)
{
	size_t remainder = size % ALIGNMENT;
	if (ALIGNMENT == 0 || remainder == 0) {
		return 0;
	} else {
		return ALIGNMENT - remainder;
	}
}

static __host__ void unit_test_deserialization_kernel(PitchedMatrix* d_pcm_sparse, PitchedMatrix* d_prm_dense)
{
	PitchedMatrix h_sparse_res;
	PitchedMatrix h_dense_res;
	CUDA_CHECK(cudaMemcpy(&h_sparse_res, d_pcm_sparse, sizeof(PitchedMatrix), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_dense_res, d_prm_dense, sizeof(PitchedMatrix), cudaMemcpyDeviceToHost));

	__half h_sparse_first_element;
	__half h_sparse_second_element;

	__half h_dense_first_element;

	CUDA_CHECK(cudaMemcpy(&h_sparse_first_element, h_sparse_res.data, sizeof(__half), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_sparse_second_element, h_sparse_res.data + 1, sizeof(__half), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(&h_dense_first_element, h_dense_res.data, sizeof(__half), cudaMemcpyDeviceToHost));

	std::cout << "Sparse [0,0]: " << __half2float(h_sparse_first_element) << "\n";
	std::cout << "Sparse [1,0]: " << __half2float(h_sparse_second_element) << "\n";

	std::cout << "Dense [0,0]: " << __half2float(h_dense_first_element) << std::endl;
}

static __global__ void deserialization_kernel(float* const h_sparse_ptr, float* const h_dense_ptr,
	PitchedMatrix* d_pcm_sparse_ptr, PitchedMatrix* d_prm_dense_ptr,
	size_t pitch, size_t rows, size_t cols)
{
	size_t thread_col = blockIdx.x * blockDim.x + threadIdx.x;
	size_t thread_row = blockIdx.y * blockDim.y + threadIdx.y;

	// NOTE: Warp divergence shouldn't happen as long as no.threads is a multiple of 32
	// I am generating neatly sized matrices. Figure out how to minimize warp divergence when matrix shape is not a multiple of 32
	if (thread_col < cols && thread_row < rows) {  // Sparse threads go here
		const float value = h_sparse_ptr[thread_col * rows + thread_row];

		__half* thread_local_ptr = reinterpret_cast<__half*>(reinterpret_cast<char*>(d_pcm_sparse_ptr->data) + thread_col * pitch);
		thread_local_ptr[thread_row] = __float2half(value);
	} else if (thread_col >= cols && thread_col < 2 * cols && thread_row >= rows && thread_row < 2 * rows) {  // Dense threads go here
		const uint32_t local_row = thread_row - rows;
		const uint32_t local_col = thread_col - cols;
		const float    value = h_dense_ptr[local_row * cols + local_col];

		__half* const thread_local_ptr = reinterpret_cast<__half*>(reinterpret_cast<char*>(d_prm_dense_ptr->data) + local_row * pitch);
		thread_local_ptr[local_col] = __float2half(value);
	}
}

static __host__ LoadBinaryOutput load_binary_into_global_mem(const std::filesystem::path& filepath)
{
	if (!std::filesystem::exists(filepath) || !std::filesystem::is_regular_file(filepath))
		THROW_RUNTIME_ERROR("Invalid file given");
	if (filepath.extension() != ".spmm")
		THROW_RUNTIME_ERROR("Invalid file type given, expected: '.spmm'");

	LoadBinaryOutput res;
	void*            host_serialized_ptr = nullptr;
	size_t           filesize = std::filesystem::file_size(filepath);

	printf("File %s with a filesize of %zu bytes will be loaded into global memory\n", filepath.c_str(), filesize);
	CUDA_CHECK(cudaMallocHost(&host_serialized_ptr, filesize));

	std::ifstream ifs(filepath, std::ios::binary);
	ifs.read(reinterpret_cast<char*>(host_serialized_ptr), filesize);

	res.rows = *reinterpret_cast<uint32_t*>(host_serialized_ptr);
	res.cols = *(reinterpret_cast<uint32_t*>(host_serialized_ptr) + 1);

	uint32_t chunk_size = sizeof(res.rows) + sizeof(res.cols);  // the bytes occupied by the first two members (rows and cols)
	uint32_t padding = calculate_padding(chunk_size);           // padding applied to the first two members (rows and cols)

	const void* const host_data_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(host_serialized_ptr) + chunk_size + padding);  // should point to the start of sparse_elements

	float* global_ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&global_ptr, filesize - (chunk_size + padding)));                                        // space just for sparse_elements and dense_elements
	CUDA_CHECK(cudaMemcpy(global_ptr, host_data_ptr, filesize - (chunk_size + padding), cudaMemcpyHostToDevice));  // copy both
	res.global_ptr = global_ptr;
	cudaFreeHost(host_serialized_ptr);

	return res;
}

/*
 * Loads the binary sparse and dense matrix into global memory,
 * deserializes them, and loads them into pitched memory.
 * Returns an SpmmInput, which is pointers to PitchedRowMajorMatrix
 * in device memory
 * WARN: THE CALLEE IS OBLIGATED TO FREE PITCHED_PTR AND D_PCM_SPARSE
 * PERF: Combine allocations and copies into a single allocation and a single merge
*/
__host__ SpmmInput deserialize(const std::filesystem::path& filepath)
{
	LoadBinaryOutput binary = load_binary_into_global_mem(filepath);
	void*            pitched_ptr = nullptr;
	size_t           pitch = 0;

	// Access with T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column
	CUDA_CHECK(cudaMallocPitch(&pitched_ptr, &pitch, binary.cols * sizeof(__half), 2 * binary.rows));

	dim3           block_size(16, 16);                                                      // 256 threads per block
	const uint32_t blocks_to_cover_cols = (binary.cols + block_size.x - 1) / block_size.x;  // ceil-ed
	const uint32_t blocks_to_cover_rows = (binary.rows + block_size.y - 1) / block_size.y;  // ceil-ed
	dim3           grid_size(2 * blocks_to_cover_cols, 2 * blocks_to_cover_rows);           // there are two of them

	size_t        chunk_size = binary.rows * binary.cols * sizeof(float);  // the size of sparse_elements in bytes
	size_t        padding = calculate_padding(chunk_size);                 // padding applied to sparse_elements
	size_t        dense_matrix_offset = chunk_size + padding;              // global_ptr+dense_matrix_offset = start of dense matrix in global memory IN BYTES
	float* const  dense_global_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(binary.global_ptr) + dense_matrix_offset);
	__half* const dense_pitched_ptr = reinterpret_cast<__half*>(reinterpret_cast<char*>(pitched_ptr) + binary.rows * pitch);

	PitchedMatrix* d_pcm_sparse = nullptr;
	PitchedMatrix* d_prm_dense = nullptr;

	PitchedMatrix h_pcm_sparse = { reinterpret_cast<__half*>(pitched_ptr), binary.rows, binary.cols, pitch };
	PitchedMatrix h_prm_dense = { dense_pitched_ptr, binary.rows, binary.cols, pitch };

	CUDA_CHECK(cudaMalloc(&d_pcm_sparse, 2 * sizeof(PitchedMatrix)));
	d_prm_dense = d_pcm_sparse + 1;

	// PERF: Merge or get rid of altogether
	CUDA_CHECK(cudaMemcpy(d_pcm_sparse, &h_pcm_sparse, sizeof(PitchedMatrix), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_prm_dense, &h_prm_dense, sizeof(PitchedMatrix), cudaMemcpyHostToDevice));

	deserialization_kernel<<<grid_size, block_size>>>(reinterpret_cast<float*>(binary.global_ptr), dense_global_ptr,
		d_pcm_sparse, d_prm_dense,
		pitch, binary.rows, binary.cols);
	CUDA_CHECK(cudaDeviceSynchronize());

	unit_test_deserialization_kernel(d_pcm_sparse, d_prm_dense);
	//
	// PERF: This should only happen once every
	// matrix needed is loaded into device memory
	// since its heavy
	CUDA_CHECK(cudaFree(binary.global_ptr));

	return { d_pcm_sparse, d_prm_dense, pitched_ptr, binary.rows, binary.cols };
}

__host__ void get_non_zero_col_predicate(PitchedMatrix* pcm_sparse, size_t rows, size_t cols)
{
	dim3         block_size(16, 16);
	const size_t blocks_to_cover_cols = (cols + block_size.x - 1) / block_size.x;
	const size_t blocks_to_cover_rows = (rows + block_size.y - 1) / block_size.y;
	dim3         grid_size(blocks_to_cover_cols, blocks_to_cover_rows);

	bool* d_predicate = nullptr;

	CUDA_CHECK(cudaMalloc(&d_predicate, cols * sizeof(bool)));
	predicate_kernel<<<grid_size, block_size>>>(pcm_sparse, d_predicate);
	CUDA_CHECK(cudaDeviceSynchronize());
	bool* h_predicate = nullptr;
	CUDA_CHECK(cudaMallocHost(&h_predicate, cols * sizeof(bool)));
	CUDA_CHECK(cudaMemcpy(h_predicate, d_predicate, cols * sizeof(bool), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < 2048; i++) {
		const bool val = h_predicate[i];
		if (val == 0) {
			printf("Found at %zu\n", i);
		}
	}
	CUDA_CHECK(cudaFree(d_predicate));
}
