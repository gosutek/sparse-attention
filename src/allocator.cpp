#include "allocator.h"

// TODO: MOVE THESE
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUSPARSE_CHECK(x)                                                                                    \
	do {                                                                                                     \
		cusparseStatus_t err = x;                                                                            \
		if (err != CUSPARSE_STATUS_SUCCESS) {                                                                \
			fprintf(stderr, "CUSPARSE error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cusparseGetErrorString(err), cusparseGetErrorName(err), err);                                \
			abort();                                                                                         \
		}                                                                                                    \
	} while (0)

void load_spmm_dlmc(SPMM<CSR>& spmm, const std::filesystem::path sparse_path, std::vector<std::filesystem::path> dense_path_list)
{
	if (!std::filesystem::exists(sparse_path) || !std::filesystem::is_regular_file(sparse_path)) {
		// TODO: Move exceptions outside of cuda files
		throw std::runtime_error("Invalid sparse path given");
	}
	// if (!std::filesystem::exists(dense_path) || !std::filesystem::is_regular_file(sparse_path)) {
	// 	// TODO: Move exceptions outside of cuda files
	// 	throw std::runtime_error("Invalid dense path given");
	// }

	// spmm.b_size = get_dlmc_byte_size(sparse_path) + get_row_major_byte_size(dense_path);
}

// TODO: Don't pass dense_path, instead you need a list of files here.
void preprocess_spmm_dlmc(SPMM<CSR>& spmm, const std::filesystem::path& sparse_path, const std::vector<std::filesystem::path>& dense_path_list)
{
	std::ifstream sparse_stream = { sparse_path };
	DLMCHeader    sparse_header = parse_dlmc_header(sparse_stream);

	// for (const auto&)
	// std::ifstream dense_stream = { dense_path };

	// RowMajorHeader dense_header = parse_row_major_header(dense_stream);

	// if (sparse_header.n_cols != dense_header.n_rows) {
	// 	// TODO: Move exceptions outside of cuda files
	// 	throw std::runtime_error("Wrong dimensions");
	// }
	//
	// spmm.b_size = get_dlmc_byte_size(sparse_header) + get_row_major_byte_size(dense_header);
}

// TODO: Move this to matrix.h
size_t get_row_major_byte_size(const RowMajorHeader& header)
{
	return header.n_rows * header.n_cols * sizeof(float);
}

// TODO: Move this to matrix.h
size_t get_dlmc_byte_size(const DLMCHeader& header)
{
	size_t row_ptr_b_size = sizeof(uint32_t) * (header.n_rows + 1);
	size_t col_idx_b_size = sizeof(uint32_t) * header.nnz;
	size_t val_b_size = sizeof(float) * header.nnz;
	//TODO: Does calc_padding_bytes() return anything other than 0?
	// Do I need this?
	return row_ptr_b_size + calc_padding_bytes(row_ptr_b_size, ALIGNMENT_BYTES) +
	       col_idx_b_size + calc_padding_bytes(col_idx_b_size, ALIGNMENT_BYTES) +
	       val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);
}

/**
 * @brief Parsing, allocation and loading of both sparse and dense
 *
 * Resulting memory block:
 *
 * +---------+---------+-----+-----+-----+
 * | row_ptr | col_idx | val | x_n | r_n |
 * +---------+---------+-----+-----+-----+
 * +-----------HOST/DEVICE---------------+
 *
 * where 'x' the dense matrices, 'r' the result matrices and 'n' = std::size(DENSE_COLS);
 *
 * 1. Allocates host space
 * 2. Generates the dense matrix and loads into host mem ~~~~~~~~~~~~~~~~~~
 * 3. Parses the sparse matrix and loads into host mem
 * 4. Copies mem block to device
 * 5. Partitions the device mem block
 */
void prepare_spmm_mem_csr(SPMM<CSR>& spmm)
{
	// Should be nullptrs
	if (spmm.host.data || spmm.dev.data) {
		// TODO: Should be a recoverable exception
		// Catch and:
		// 1. If condition is exclusive throw a std::runtime_error
		// 2. If both are allocated then just warn and continue
		std::cout << "SPMM handle has been allocated/misallocated prior to the calling of this function (prepare_spmm_csr)";
		return;
	}

	spmm.host.data = cuda_malloc_host(spmm.b_size);
	spmm.host.s = parse_dlmc(spmm.host.data, spmm.sparse_path);

	void* start_of_dense = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(spmm.host.s.row_ptr) + spmm.host.s.b_size);  // at the start of the sparse matrix, skip spmm.host.s.b_size bytes.
	spmm.host.d[0] = reinterpret_cast<float*>(start_of_dense);
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCH_DIMS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCH_DIMS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCH_DIMS[i] * MAT_SIZE;
		}
	}

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_dense) + BENCH_DIMS[std::size(BENCH_DIMS) - 1] * MAT_SIZE;  // skip the dense matrix

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}

	spmm.dev.data = cuda_malloc_device(spmm.b_size);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCH_DIMS_BSIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSR(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr += spmm.host.s.b_size;

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.dev.d[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.dev.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}
}

void prepare_spmm_csc(SPMM<CSC>& spmm)
{
	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
	}

	std::ifstream file_stream = { spmm.sparse_path };
	DLMCHeader    header = parse_dlmc_header(file_stream);

	size_t col_ptr_b_size = sizeof(uint32_t) * (header.n_cols + 1);
	size_t row_idx_b_size = sizeof(uint32_t) * header.nnz;
	size_t val_b_size = sizeof(float) * header.nnz;
	size_t sparse_b_size_aligned = col_ptr_b_size + calc_padding_bytes(col_ptr_b_size, ALIGNMENT_BYTES) +
	                               row_idx_b_size + calc_padding_bytes(row_idx_b_size, ALIGNMENT_BYTES) +
	                               val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);

	/**
    * Twice the total size of the dense matrices.
    * Once for the input
    * Twice for the result
    **/
	spmm.b_size = sparse_b_size_aligned + 2 * BENCH_DIMS_BSIZE;
	spmm.host.data = cuda_malloc_host(spmm.b_size);
	spmm.host.d[0] = reinterpret_cast<float*>(spmm.host.data);

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCH_DIMS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCH_DIMS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCH_DIMS[i] * MAT_SIZE;
		}
	}

	assert((reinterpret_cast<uintptr_t>(spmm.host.d[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	void* start_of_sparse = spmm.host.d[std::size(BENCH_DIMS) - 1] +           // from the last ptr of spmm.host.d
	                        BENCH_DIMS[std::size(BENCH_DIMS) - 1] * MAT_SIZE;  // skip 512 * 512 floats

	// start_of_sparse is 128-byte aligned guaranteed
	spmm.host.s = parse_csc_dlmc(start_of_sparse, spmm.sparse_path);
	spmm.host.s.max_nnz_per_col = calc_max_nnz_per_col(spmm.host.s);

	assert((reinterpret_cast<uintptr_t>(spmm.host.s.col_ptr) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.row_idx) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.val) & (ALIGNMENT_BYTES - 1)) == 0);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	// WARN: asserts cost
	assert(sparse_b_size_aligned == spmm.host.s.b_size);

	/*
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
      * | x_32 | x_64 | x_128 | x_256 | x_512 | col_ptr | row_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
      * +------------------------------------------HOST/DEVICE------------------------------------------------+
   */

	spmm.dev.data = cuda_malloc_device(spmm.b_size);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCH_DIMS_BSIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.dev.d[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSC(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr += spmm.host.s.b_size;

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.dev.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}
}

bool warmup_spmm_csr(SPMM<CSR>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSR>&, const uint32_t))
{
	const size_t res_size = BENCH_DIMS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemset(spmm.dev.r[size_idx], 0.0f, res_size * sizeof(float)));
	// PERF: Bounds check
	assert(size_idx < std::size(BENCH_DIMS) - 1);
	run_kernel(spmm, size_idx);

	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csr(spmm, cusparse);

	CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	return verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
}

void prepare_cusparse_csr(SPMM<CSR>& spmm, CuSparse& cusparse)
{
	if (!spmm.host.data || !spmm.dev.data) {
		throw std::runtime_error("prepare_cusparse_csr() received a unallocated SPMM<CSR>&");
	}
	CUSPARSE_CHECK(cusparseCreateCsr(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], spmm.dev.s.cols, BENCH_DIMS[i], spmm.dev.s.cols, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_COL));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.rows, BENCH_DIMS[i], BENCH_DIMS[i], spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

void prepare_cusparse_csc(SPMM<CSC>& spmm, CuSparse& cusparse)
{
	CUSPARSE_CHECK(cusparseCreateCsc(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], BENCH_DIMS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCH_DIMS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

// void prepare_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	// mhsa_load_host_csc(mhsa, mhsa.config, mhsa.dlmc, mhsa.weights);
//
// 	// TODO: Find a better name
// 	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
// 	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;
//
// 	size_t res_b_size = sizeof(float) * (kv_size * 4 + gemm_res_size * 2 + 1);  // Q, K, V, gemm result, float acc for softmax, Attention matrix, Final Result
//
// 	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
// 	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));
//
// 	/*
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * | x | w_q | w_k | w_v | w_o | mask | Q | K | V | QK^T | ACC | A | Final Result |
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * +-------------HOST-----------------+----------------DEVICE---------------------+
//    */
//
// 	res.x = reinterpret_cast<float*>(mhsa.dev);
// 	size_t b_x_size = sizeof(float) * kv_size;
//
// 	char* ptr = reinterpret_cast<char*>(res.x) + b_x_size;
//
// 	// TODO: This call copy assignment operator of CSC
// 	// check if the custom one does what you want
// 	res.w_q = mhsa.weights.w_q[0];
// 	res.w_q.partition(ptr);
// 	ptr += res.w_q.b_size;
//
// 	res.w_k = mhsa.weights.w_k[0];
// 	res.w_k.partition(ptr);
// 	ptr += res.w_k.b_size;
//
// 	res.w_v = mhsa.weights.w_v[0];
// 	res.w_v.partition(ptr);
// 	ptr += res.w_v.b_size;
//
// 	res.w_o = mhsa.weights.w_o[0];
// 	res.w_o.partition(ptr);
// 	ptr += res.w_o.b_size;
//
// 	res.q_res = reinterpret_cast<float*>(ptr);
// 	res.k_res = res.q_res + kv_size;
// 	res.v_res = res.k_res + kv_size;
// 	res.gemm_res = res.v_res + kv_size;
// 	res.softmax_acc = res.gemm_res + gemm_res_size;
// 	res.attention = res.softmax_acc + 1;
//
// 	return res;
// }
//
// void run_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	DevMHSA      d = prepare_mhsa(mhsa);
// 	const size_t m = mhsa.config.input_sequence_size;
// 	const size_t n = d.w_q.cols;
//
// 	// One thread per element of the output
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_gm(32, 32);
// 	dim3 spmm_grid_gm(
// 		(n + spmm_block_gm.x - 1) / spmm_block_gm.x,
// 		(m + spmm_block_gm.y - 1) / spmm_block_gm.y);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_sm(512);
// 	dim3 spmm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x32)=(32x32)
// 	dim3 gemm_block_sm(32);
// 	dim3 gemm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x32)
// 	dim3 softmax_block(32, 32);
// 	dim3 softmax_grid(
// 		(m + softmax_block.x - 1) / softmax_block.x,
// 		(m + softmax_block.y - 1) / softmax_block.y);  // This should actually be equal to (1,1) i.e. one block
//
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_q.col_ptr, d.w_q.row_idx, d.w_q.val, mhsa.config.input_sequence_size, d.w_q.rows, d.w_q.cols, d.q_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_k.col_ptr, d.w_k.row_idx, d.w_k.val, mhsa.config.input_sequence_size, d.w_k.rows, d.w_k.cols, d.k_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_v.col_ptr, d.w_v.row_idx, d.w_v.val, mhsa.config.input_sequence_size, d.w_v.rows, d.w_v.cols, d.v_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	gemm<<<gemm_grid_sm, gemm_block_sm>>>(d.q_res, d.k_res, mhsa.config.input_sequence_size, d.w_q.rows, mhsa.config.input_sequence_size, d.gemm_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	softmax<<<softmax_grid, softmax_block>>>(d.gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, d.softmax_acc, d.attention);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// TODO: can this be async?
// 	// TODO: THIS NEEDS TO WRITE TO PAGE-LOCKED MEMORY NOT SOME RANDOM ALLOCATED MEMORY
// 	//
// 	// CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
// }
