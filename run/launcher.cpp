#include "launcher.h"

void run_mhsa(MHSA<CSC, CSR>& mhsa)
{
	DevMHSA      d = prepare_mhsa(mhsa);
	const size_t m = mhsa.config.input_sequence_size;
	const size_t n = d.w_q.cols;

	// One thread per element of the output
	// One thread block per 32x32 submatrix of the output
	// (32x512)*(512x512)=(32x512)
	dim3 spmm_block_gm(32, 32);
	dim3 spmm_grid_gm(
		(n + spmm_block_gm.x - 1) / spmm_block_gm.x,
		(m + spmm_block_gm.y - 1) / spmm_block_gm.y);

	// One thread per element of the output.
	// One thread block stretched across a row of the output
	// (32x512)*(512x512)=(32x512)
	dim3 spmm_block_sm(512);
	dim3 spmm_grid_sm(32);

	// One thread per element of the output.
	// One thread block stretched across a row of the output
	// (32x512)*(512x32)=(32x32)
	dim3 gemm_block_sm(32);
	dim3 gemm_grid_sm(32);

	// One thread per element of the output.
	// One thread block per 32x32 submatrix of the output
	// (32x32)
	dim3 softmax_block(32, 32);
	dim3 softmax_grid(
		(m + softmax_block.x - 1) / softmax_block.x,
		(m + softmax_block.y - 1) / softmax_block.y);  // This should actually be equal to (1,1) i.e. one block

	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_q.col_ptr, d.w_q.row_idx, d.w_q.val, mhsa.config.input_sequence_size, d.w_q.rows, d.w_q.cols, d.q_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_k.col_ptr, d.w_k.row_idx, d.w_k.val, mhsa.config.input_sequence_size, d.w_k.rows, d.w_k.cols, d.k_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_v.col_ptr, d.w_v.row_idx, d.w_v.val, mhsa.config.input_sequence_size, d.w_v.rows, d.w_v.cols, d.v_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	gemm<<<gemm_grid_sm, gemm_block_sm>>>(d.q_res, d.k_res, mhsa.config.input_sequence_size, d.w_q.rows, mhsa.config.input_sequence_size, d.gemm_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	softmax<<<softmax_grid, softmax_block>>>(d.gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, d.softmax_acc, d.attention);

	CUDA_CHECK(cudaDeviceSynchronize());

	// TODO: can this be async?
	// TODO: THIS NEEDS TO WRITE TO PAGE-LOCKED MEMORY NOT SOME RANDOM ALLOCATED MEMORY
	//
	// CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
}
void prepare_mhsa(MHSA<CSC, CSR>& mhsa)
{
	// mhsa_load_host_csc(mhsa, mhsa.config, mhsa.dlmc, mhsa.weights);

	// TODO: Find a better name
	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;

	size_t res_b_size = sizeof(float) * (kv_size * 4 + gemm_res_size * 2 + 1);  // Q, K, V, gemm result, float acc for softmax, Attention matrix, Final Result

	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));

	/*
      * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
      * | x | w_q | w_k | w_v | w_o | mask | Q | K | V | QK^T | ACC | A | Final Result |
      * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
      * +-------------HOST-----------------+----------------DEVICE---------------------+
   */

	res.x = reinterpret_cast<float*>(mhsa.dev);
	size_t b_x_size = sizeof(float) * kv_size;

	char* ptr = reinterpret_cast<char*>(res.x) + b_x_size;

	// TODO: This call copy assignment operator of CSC
	// check if the custom one does what you want
	res.w_q = mhsa.weights.w_q[0];
	res.w_q.partition(ptr);
	ptr += res.w_q.b_size;

	res.w_k = mhsa.weights.w_k[0];
	res.w_k.partition(ptr);
	ptr += res.w_k.b_size;

	res.w_v = mhsa.weights.w_v[0];
	res.w_v.partition(ptr);
	ptr += res.w_v.b_size;

	res.w_o = mhsa.weights.w_o[0];
	res.w_o.partition(ptr);
	ptr += res.w_o.b_size;

	res.q_res = reinterpret_cast<float*>(ptr);
	res.k_res = res.q_res + kv_size;
	res.v_res = res.k_res + kv_size;
	res.gemm_res = res.v_res + kv_size;
	res.softmax_acc = res.gemm_res + gemm_res_size;
	res.attention = res.softmax_acc + 1;

	return res;
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

	void* start_of_sparse = spmm.host.d[std::size(BENCH_DIMS) - 1] +           // from the last ptr of spmm.host.d
	                        BENCH_DIMS[std::size(BENCH_DIMS) - 1] * MAT_SIZE;  // skip 512 * 512 floats

	// start_of_sparse is 128-byte aligned guaranteed
	spmm.host.s = parse_csc_dlmc(start_of_sparse, spmm.sparse_path);
	spmm.host.s.max_nnz_per_col = calc_max_nnz_per_col(spmm.host.s);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(float);
	}

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
	spmm.dev.s = Csr(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
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

bool warmup_spmm_csc(SPMM<CSC>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSC>&, const uint32_t))
{
	const size_t res_size = BENCH_DIMS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemset(spmm.dev.r[size_idx], 0.0f, res_size * sizeof(float)));
	// PERF: Bounds check
	assert(size_idx < std::size(BENCH_DIMS) - 1);  // DON'T REMOVE, YOU ARE DOING size_idx + 1 later
	run_kernel(spmm, size_idx);

	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csc(spmm, cusparse);

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

void run_spmm_naive_elemwise_gmem(SPMM<CSR>& spmm, const uint32_t idx)
{
	const size_t m = spmm.dev.s.rows;
	const size_t k = spmm.dev.s.cols;
	const size_t n = BENCH_DIMS[idx];

	constexpr size_t BM = 8;
	constexpr size_t BK = BM;

	assert(BM <= 32);  // otherwise threads per block exceed max
	dim3 grid(CEIL_DIV(MAT_SIZE, BK), CEIL_DIV(m, BM));
	dim3 block(BK, BM);

	spmm_naive_elemwise_gmem<<<grid, block>>>(spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val, spmm.dev.d[idx], m, k, n, spmm.dev.r[idx]);
}

void run_spmm_naive_elemwise_csc_gmem(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t BN = 8;
	constexpr size_t BK = BN;

	assert(BN <= 32);  // otherwise threads per block exceed max
	dim3 grid(CEIL_DIV(MAT_SIZE, BN), CEIL_DIV(m, BK));
	dim3 block(BN, BK);

	spmm_naive_elemwise_csc_gmem<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_naive_elemwise_csc_smem(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	dim3 grid(m);
	dim3 block(n);

	spmm_naive_elemwise_csc_smem<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_coalesced_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	dim3 grid(MAT_SIZE);
	dim3 block(128);

	spmm_coalesced_elemwise_csr<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

// void run_spmm_blocktiling_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx)
// {
// 	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
// 	const size_t k = spmm.dev.s.rows;
// 	const size_t n = spmm.dev.s.cols;
//
// 	constexpr size_t BN = 256;
// 	constexpr size_t TN = 4;
//
// 	dim3 grid(m, n / BN);
// 	dim3 block(CEIL_DIV(BN, TN));
//
// 	spmm_blocktiling_elemwise_csr<<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
// }

void run_spmm_coalesced_nnzwise(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 64;

	dim3 grid(n, m);
	dim3 block(n_threads);

	spmm_coalesced_nnzwise<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_coalesced_nnzwise_no_smem(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 64;

	dim3 grid(n, m);
	dim3 block(n_threads);

	spmm_coalesced_nnzwise_no_smem<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

void run_spmm_coalesced_nnzwise_last(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 32;
	constexpr size_t bn = 16;

	dim3 grid(CEIL_DIV(n, bn), m);
	dim3 block(n_threads);

	spmm_coalesced_nnzwise_last<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, bn, spmm.dev.r[idx]);
}

void run_spmm_vectorized_nnzwise_regs(SPMM<CSC>& spmm, const uint32_t idx)
{
	const size_t m = BENCH_DIMS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	constexpr size_t n_threads = 32;
	constexpr size_t BK = 512;

	dim3 grid(n, m, CEIL_DIV(MAT_SIZE, BK));
	dim3 block(n_threads);

	spmm_vectorized_nnzwise_regs<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, spmm.dev.r[idx]);
}

std::vector<float> read_row_major_from_rm(const std::filesystem::path& filepath, size_t size)
{
	if (!std::filesystem::exists(filepath) && !std::filesystem::is_regular_file(filepath)) {
		throw std::runtime_error(filepath.string() + " does not exist\n");
	}
	std::vector<float> res;
	res.reserve(size);

	std::ifstream file_stream(filepath, std::ios_base::in);
	if (!file_stream) {
		throw std::runtime_error("Failed to open file:" + filepath.string());
	}
	float tmp;
	while (file_stream >> tmp) {
		res.push_back(tmp);
	}
	return res;
}

Csr::Matrix parse_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		// TODO: Remove exceptions
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DlmcHeader header = parse_dlmc_header(file_stream);

	Csr::Matrix csr;
	Csr::init(csr, header.rows, header.cols, header.nnz);
	Csr::partition(csr, reinterpret_cast<uintptr_t>(dst));

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < csr.row_ptr_count; ++i) {
		row_ptr_stream >> token;
		csr.row_ptr[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::getline(file_stream, line);
	std::istringstream col_idx_stream(line);
	for (size_t i = 0; i < csr.col_idx_count; ++i) {
		col_idx_stream >> token;
		csr.col_idx[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < csr.val_count; ++i) {
		csr.val[i] = uni_real_dist(rng);
	}

	return csr;
}

Csc::Matrix parse_csc_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		// TODO: Remove exceptions
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DlmcHeader  header = parse_dlmc_header(file_stream);
	Csc::Matrix csc;
	Csc::init(csc, header.rows, header.cols, header.nnz);
	Csc::partition(csc, reinterpret_cast<uintptr_t>(dst));

	std::vector<uint32_t> row_ptr_vec(header.rows + 1, 0);

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < header.rows + 1; ++i) {
		row_ptr_stream >> token;
		row_ptr_vec[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::vector<uint32_t> col_idx_vec(header.nnz, 0);

	std::getline(file_stream, line);
	std::istringstream col_idx_stream(line);
	for (size_t i = 0; i < header.nnz; ++i) {
		col_idx_stream >> token;
		col_idx_vec[i] = static_cast<uint32_t>(std::stoi(token));
	}

	csr_to_csc(csc, row_ptr_vec, col_idx_vec);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < csc.val_count; ++i) {
		csc.val[i] = uni_real_dist(rng);
	}

	return csc;
}

std::vector<std::filesystem::path> collect_rec_input(const std::filesystem::path& path)
{
	uint32_t                                            n_unknown_extention_files{};
	std::vector<std::filesystem::path>                  input_files;
	const std::filesystem::recursive_directory_iterator rec_dir_iter(path);

	for (const std::filesystem::path& path : rec_dir_iter) {
		if (std::filesystem::is_regular_file(path) && path.extension() == ".smtx") {
			input_files.push_back(path);
		} else {
			n_unknown_extention_files++;
		}
	}
	std::cout << std::format("Found in directory '{}':\n", path.string())
			  << std::format("\t- {} '.smtx' file(s)\n", input_files.size())
			  << std::format("\t- {} unsupported file(s)\n", n_unknown_extention_files);

	return input_files;
}

/*
 * a(m, k)
 * b(k, n)
 * c(m, n)
 * Expects b to be in column-major
 */
[[maybe_unused]] static std::vector<float> host_spmm_rm_cm(const std::vector<float>& a, const std::vector<float>& b, size_t m, size_t k, size_t n)
{
	std::vector<float> res;
	res.reserve(m * n);

	for (size_t a_row = 0; a_row < m; ++a_row) {
		for (size_t b_col = 0; b_col < n; ++b_col) {
			float acc = 0;
			for (size_t i = 0; i < k; ++i) {
				acc += a[a_row * k + i] * b[b_col * k + i];
			}
			res.push_back(acc);
		}
	}

	return res;
}

[[maybe_unused]] static std::vector<float> host_spmm_rm_rm(std::vector<float> a, std::vector<float> b, size_t m, size_t k, size_t n)
{
	std::vector<float> res;
	res.reserve(m * n);

	for (size_t a_row = 0; a_row < m; ++a_row) {
		for (size_t b_col = 0; b_col < n; ++b_col) {
			float acc = 0;
			for (size_t i = 0; i < k; ++i) {
				acc += a[a_row * k + i] * b[i * k + b_col];
			}
			res.push_back(acc);
		}
	}

	return res;
}

bool verify_res(const float* const actual, const float* const expected, size_t n)
{
	double diff = 0.0;
	for (size_t i = 0; i < n; ++i) {
		diff = std::fabs(actual[i] - expected[i]);
		// std::cout << std::format(
		// 	"Actual: {}, Expected: {}, Diff: {}, Pos: {}\n", actual[i], expected[i], diff, i);
		if (std::isnan(diff) || diff > 0.01) {
			std::cout << std::format(
				"Values diverge -> Actual: {}, Expected: {} (Diff {:.4f}), pos: {:d}\n",
				actual[i], expected[i], diff, i);
			return false;
		}
	}
	return true;
}
