#include "launcher.h"

SpmmContext setup_spmm(const ExecutionContext_t handle, const std::filesystem::path& sp_path)
{
	CSR          h_csr = parse_csr_dlmc(sp_path);
	SpMatDescr_t d_csr = NULL;

	CHECK_SPMM(sp_csr_create(handle, &d_csr, h_csr.rows, h_csr.cols, h_csr.nnz, h_csr.row_ptr.data(), h_csr.col_idx.data(), h_csr.val.data()));

	std::vector<f32> h_dn;
	gen_synth_weights_vec(h_dn, h_csr.rows * h_csr.cols);

	DnMatDescr_t d_dn = NULL;
	CHECK_SPMM(dn_cm_create(handle, &d_dn, h_csr.cols, h_csr.cols, h_dn.data()));

	std::vector<f32> h_res(h_csr.rows * h_csr.cols, 0);

	DnMatDescr_t d_res = NULL;
	CHECK_SPMM(dn_rm_create(handle, &d_res, h_csr.rows, h_csr.cols, h_res.data()));

	return {
		.h_csr = std::move(h_csr),
		.d_csr = d_csr,
		.h_dn = std::move(h_dn),
		.d_dn = d_dn,
		.h_res = std::move(h_res),
		.d_res = d_res
	};
}

CusparseContext setup_cusparse(const cusparseHandle_t handle, const SpMatDescr_t d_sp, const DnMatDescr_t d_dn, const DnMatDescr_t d_res)
{
	constexpr const f32 alpha = 1.0f;
	constexpr const f32 beta = 0.0f;

	u32  rows, cols, nnz, *row_ptr, *col_idx;
	f32* val;

	CHECK_SPMM(sp_csr_get(d_sp, &rows, &cols, &nnz, &row_ptr, &col_idx, &val));

	// TODO: Check the ptr is on the device
	cusparseSpMatDescr_t cusparse_csr = NULL;
	CHECK_CUSPARSE(cusparseCreateCsr(&cusparse_csr,
		rows, cols, nnz,
		row_ptr, col_idx, val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	CHECK_SPMM(dn_cm_get(d_dn, &rows, &cols, &val));
	// TODO: Check the ptr is on the device
	cusparseDnMatDescr_t cusparse_dn = NULL;
	CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_dn,
		rows, cols, cols, val, CUDA_R_32F, CUSPARSE_ORDER_COL));

	// TODO: Check the ptr is on the device
	CHECK_SPMM(dn_rm_get(d_res, &rows, &cols, &val));
	std::vector<f32>     h_res(rows * cols, 0);
	cusparseDnMatDescr_t cusparse_res = NULL;
	CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_res, rows, cols, cols, val, CUDA_R_32F, CUSPARSE_ORDER_ROW));

	u64 buffer_size;
	CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, cusparse_csr, cusparse_dn, &beta, cusparse_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buffer_size));

	void* cusparse_buffer = nullptr;
	CHECK_CUDA(cudaMalloc(&cusparse_buffer, buffer_size));
	CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, cusparse_csr, cusparse_dn, &beta, cusparse_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse_buffer));

	return {
		.d_csr = cusparse_csr,
		.d_dn = cusparse_dn,
		.h_res = std::move(h_res),
		.d_res = cusparse_res,
		.buffer_size = buffer_size,
		.buffer = cusparse_buffer,
		.alpha = alpha,
		.beta = beta
	};
}

// void prepare_cusparse_csc(SPMM<CSC>& spmm, CuSparse& cusparse)
// {
// 	CHECK_CUSPARSE(cusparseCreateCsc(&cusparse.sparse,
// 		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
// 		spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
// 		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
//
// 	size_t tmp = 0;
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse.dense[i], BENCH_DIMS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
// 		CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCH_DIMS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));
//
// 		CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparse.handle,
// 			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
// 			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
// 			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &tmp));
//
// 		cusparse.work_buffer_size += tmp;
// 	}
//
// 	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
// 	if (!cusparse.work_buffer) {
// 		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
// 	}
// }
//
// bool warmup_spmm_csr(SPMM<CSR>& spmm, const u32 size_idx, void (*run_kernel)(SPMM<CSR>&, const u32))
// {
// 	const size_t res_size = BENCH_DIMS[size_idx] * MAT_SIZE;
// 	CHECK_CUDA(cudaMemset(spmm.dev.r[size_idx], 0.0f, res_size * sizeof(f32)));
// 	// PERF: Bounds check
// 	assert(size_idx < std::size(BENCH_DIMS) - 1);
// 	run_kernel(spmm, size_idx);
//
// 	CHECK_CUDA(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(f32), cudaMemcpyDeviceToHost));
// 	CHECK_CUDA(cudaDeviceSynchronize());
//
// 	// WARN: Temporary hack
// 	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(f32));
//
// 	CuSparse cusparse;
// 	cusparseCreate(&cusparse.handle);
// 	prepare_cusparse_csr(spmm, cusparse);
//
// 	CHECK_CUSPARSE(cusparseSpMM(cusparse.handle,
// 		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
// 		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
// 	CHECK_CUDA(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(f32), cudaMemcpyDeviceToHost));
//
// 	cuda_dealloc_device(cusparse.work_buffer);
//
// 	cusparseDestroySpMat(cusparse.sparse);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		cusparseDestroyDnMat(cusparse.dense[i]);
// 		cusparseDestroyDnMat(cusparse.res[i]);
// 	}
// 	cusparseDestroy(cusparse.handle);
//
// 	return verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
// }
//
// void prepare_cusparse_csr(SPMM<CSR>& spmm, CuSparse& cusparse)
// {
// 	if (!spmm.host.data || !spmm.dev.data) {
// 		throw std::runtime_error("prepare_cusparse_csr() received a unallocated SPMM<CSR>&");
// 	}
// 	CHECK_CUSPARSE(cusparseCreateCsr(&cusparse.sparse,
// 		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
// 		spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val,
// 		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
//
// 	size_t tmp = 0;
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse.dense[i], spmm.dev.s.cols, BENCH_DIMS[i], spmm.dev.s.cols, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_COL));
// 		CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.rows, BENCH_DIMS[i], BENCH_DIMS[i], spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
//
// 		CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparse.handle,
// 			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
// 			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
// 			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &tmp));
//
// 		cusparse.work_buffer_size += tmp;
// 	}
//
// 	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
// 	if (!cusparse.work_buffer) {
// 		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
// 	}
// }
// void prepare_spmm_csc(SPMM<CSC>& spmm)
// {
// 	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
// 		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
// 	}
//
// 	std::ifstream file_stream = { spmm.sparse_path };
// 	DLMCHeader    header = parse_dlmc_header(file_stream);
//
// 	size_t col_ptr_b_size = sizeof(u32) * (header.n_cols + 1);
// 	size_t row_idx_b_size = sizeof(u32) * header.nnz;
// 	size_t val_b_size = sizeof(f32) * header.nnz;
// 	size_t sparse_b_size_aligned = col_ptr_b_size + calc_padding_bytes(col_ptr_b_size, ALIGNMENT_BYTES) +
// 	                               row_idx_b_size + calc_padding_bytes(row_idx_b_size, ALIGNMENT_BYTES) +
// 	                               val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);
//
// 	/**
//     * Twice the total size of the dense matrices.
//     * Once for the input
//     * Twice for the result
//     **/
// 	spmm.b_size = sparse_b_size_aligned + 2 * BENCH_DIMS_BSIZE;
// 	spmm.host.data = cuda_malloc_host(spmm.b_size);
// 	spmm.host.d[0] = reinterpret_cast<f32*>(spmm.host.data);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		generate_token_embeddings(spmm.host.d[i], BENCH_DIMS[i] * MAT_SIZE);
// 		if (i + 1 < std::size(BENCH_DIMS)) {
// 			spmm.host.d[i + 1] = spmm.host.d[i] + BENCH_DIMS[i] * MAT_SIZE;
// 		}
// 	}
//
// 	void* start_of_sparse = spmm.host.d[std::size(BENCH_DIMS) - 1] +           // from the last ptr of spmm.host.d
// 	                        BENCH_DIMS[std::size(BENCH_DIMS) - 1] * MAT_SIZE;  // skip 512 * 512 floats
//
// 	// start_of_sparse is 128-byte aligned guaranteed
// 	spmm.host.s = parse_csc_dlmc(start_of_sparse, spmm.sparse_path);
// 	spmm.host.s.max_nnz_per_col = calc_max_nnz_per_col(spmm.host.s);
//
// 	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;
//
// 	// TODO: use uintptr_t instead of pointer arithmetic on f32* (??)
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.host.r[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
//
// 	// WARN: asserts cost
// 	assert(sparse_b_size_aligned == spmm.host.s.b_size);
//
// 	/*
//       * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
//       * | x_32 | x_64 | x_128 | x_256 | x_512 | col_ptr | row_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
//       * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
//       * +------------------------------------------HOST/DEVICE------------------------------------------------+
//    */
//
// 	spmm.dev.data = cuda_malloc_device(spmm.b_size);
// 	CHECK_CUDA(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCH_DIMS_BSIZE, cudaMemcpyHostToDevice));
//
// 	// Partition dev
// 	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.dev.d[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
//
// 	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
// 	spmm.dev.s = CSC(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
// 	spmm.dev.s.partition(ptr);
//
// 	ptr += spmm.host.s.b_size;
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.dev.r[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
// }
// /**
//  * @brief Parsing, allocation and loading of both sparse and dense
//  *
//  * Resulting memory block:
//  *
//  * +---------+---------+-----+-----+-----+
//  * | row_ptr | col_idx | val | x_n | r_n |
//  * +---------+---------+-----+-----+-----+
//  * +-----------HOST/DEVICE---------------+
//  *
//  * where 'x' the dense matrices, 'r' the result matrices and 'n' = std::size(DENSE_COLS);
//  *
//  * 1. Allocates host space
//  * 2. Generates the dense matrix and loads into host mem ~~~~~~~~~~~~~~~~~~
//  * 3. Parses the sparse matrix and loads into host mem
//  * 4. Copies mem block to device
//  * 5. Partitions the device mem block
//  */
// void prepare_spmm_mem_csr(SPMM<CSR>& spmm)
// {
// 	// Should be nullptrs
// 	if (spmm.host.data || spmm.dev.data) {
// 		// TODO: Should be a recoverable exception
// 		// Catch and:
// 		// 1. If condition is exclusive throw a std::runtime_error
// 		// 2. If both are allocated then just warn and continue
// 		std::cout << "SPMM handle has been allocated/misallocated prior to the calling of this function (prepare_spmm_csr)";
// 		return;
// 	}
//
// 	spmm.host.data = cuda_malloc_host(spmm.b_size);
// 	spmm.host.s = parse_dlmc(spmm.host.data, spmm.sparse_path);
//
// 	void* start_of_dense = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(spmm.host.s.row_ptr) + spmm.host.s.b_size);  // at the start of the sparse matrix, skip spmm.host.s.b_size bytes.
// 	spmm.host.d[0] = reinterpret_cast<f32*>(start_of_dense);
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		generate_token_embeddings(spmm.host.d[i], BENCH_DIMS[i] * MAT_SIZE);
// 		if (i + 1 < std::size(BENCH_DIMS)) {
// 			spmm.host.d[i + 1] = spmm.host.d[i] + BENCH_DIMS[i] * MAT_SIZE;
// 		}
// 	}
//
// 	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_dense) + BENCH_DIMS[std::size(BENCH_DIMS) - 1] * MAT_SIZE;  // skip the dense matrix
//
// 	// TODO: use uintptr_t instead of pointer arithmetic on f32* (??)
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.host.r[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
//
// 	spmm.dev.data = cuda_malloc_device(spmm.b_size);
// 	CHECK_CUDA(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCH_DIMS_BSIZE, cudaMemcpyHostToDevice));
//
// 	// Partition dev
// 	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);
//
// 	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
// 	spmm.dev.s = Csr(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
// 	spmm.dev.s.partition(ptr);
//
// 	ptr += spmm.host.s.b_size;
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.dev.d[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		spmm.dev.r[i] = reinterpret_cast<f32*>(ptr);
// 		ptr += BENCH_DIMS[i] * MAT_SIZE * sizeof(f32);
// 	}
// }
//
// // TODO: Don't pass dense_path, instead you need a list of files here.
// void preprocess_spmm_dlmc(SPMM<CSR>& spmm, const std::filesystem::path& sparse_path, const std::vector<std::filesystem::path>& dense_path_list)
// {
// 	std::ifstream sparse_stream = { sparse_path };
// 	DLMCHeader    sparse_header = parse_dlmc_header(sparse_stream);
//
// 	// for (const auto&)
// 	// std::ifstream dense_stream = { dense_path };
//
// 	// RowMajorHeader dense_header = parse_row_major_header(dense_stream);
//
// 	// if (sparse_header.n_cols != dense_header.n_rows) {
// 	// 	// TODO: Move exceptions outside of cuda files
// 	// 	throw std::runtime_error("Wrong dimensions");
// 	// }
// 	//
// 	// spmm.b_size = get_dlmc_byte_size(sparse_header) + get_row_major_byte_size(dense_header);
// }
//
// void load_spmm_dlmc(SPMM<CSR>& spmm, const std::filesystem::path sparse_path, std::vector<std::filesystem::path> dense_path_list)
// {
// 	if (!std::filesystem::exists(sparse_path) || !std::filesystem::is_regular_file(sparse_path)) {
// 		// TODO: Move exceptions outside of cuda files
// 		throw std::runtime_error("Invalid sparse path given");
// 	}
// 	// if (!std::filesystem::exists(dense_path) || !std::filesystem::is_regular_file(sparse_path)) {
// 	// 	// TODO: Move exceptions outside of cuda files
// 	// 	throw std::runtime_error("Invalid dense path given");
// 	// }
//
// 	// spmm.b_size = get_dlmc_byte_size(sparse_path) + get_row_major_byte_size(dense_path);
// }
//
// bool warmup_spmm_csc(SPMM<CSC>& spmm, const u32 size_idx, void (*run_kernel)(SPMM<CSC>&, const u32))
// {
// 	const size_t res_size = BENCH_DIMS[size_idx] * MAT_SIZE;
// 	CHECK_CUDA(cudaMemset(spmm.dev.r[size_idx], 0.0f, res_size * sizeof(f32)));
// 	// PERF: Bounds check
// 	assert(size_idx < std::size(BENCH_DIMS) - 1);  // DON'T REMOVE, YOU ARE DOING size_idx + 1 later
// 	run_kernel(spmm, size_idx);
//
// 	CHECK_CUDA(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(f32), cudaMemcpyDeviceToHost));
// 	CHECK_CUDA(cudaDeviceSynchronize());
//
// 	// WARN: Temporary hack
// 	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(f32));
//
// 	CuSparse cusparse;
// 	cusparseCreate(&cusparse.handle);
// 	prepare_cusparse_csc(spmm, cusparse);
//
// 	CHECK_CUSPARSE(cusparseSpMM(cusparse.handle,
// 		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
// 		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
// 	CHECK_CUDA(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(f32), cudaMemcpyDeviceToHost));
//
// 	cuda_dealloc_device(cusparse.work_buffer);
//
// 	cusparseDestroySpMat(cusparse.sparse);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		cusparseDestroyDnMat(cusparse.dense[i]);
// 		cusparseDestroyDnMat(cusparse.res[i]);
// 	}
// 	cusparseDestroy(cusparse.handle);
//
// 	return verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
// }
//
// void run_spmm_coalesced_nnzwise_last(SPMM<CSC>& spmm, const u32 idx)
// {
// 	const size_t m = BENCH_DIMS[idx];
// 	const size_t k = spmm.dev.s.rows;
// 	const size_t n = spmm.dev.s.cols;
//
// 	constexpr size_t n_threads = 32;
// 	constexpr size_t bn = 16;
//
// 	dim3 grid(CEIL_DIV(n, bn), m);
// 	dim3 block(n_threads);
//
// 	spmm_coalesced_nnzwise_last<n_threads><<<grid, block>>>(spmm.dev.d[idx], spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val, m, k, n, bn, spmm.dev.r[idx]);
// }
//
// std::vector<f32> read_row_major_from_rm(const std::filesystem::path& filepath, size_t size)
// {
// 	if (!std::filesystem::exists(filepath) && !std::filesystem::is_regular_file(filepath)) {
// 		throw std::runtime_error(filepath.string() + " does not exist\n");
// 	}
// 	std::vector<f32> res;
// 	res.reserve(size);
//
// 	std::ifstream file_stream(filepath, std::ios_base::in);
// 	if (!file_stream) {
// 		throw std::runtime_error("Failed to open file:" + filepath.string());
// 	}
// 	f32 tmp;
// 	while (file_stream >> tmp) {
// 		res.push_back(tmp);
// 	}
// 	return res;
// }
//
// std::vector<std::filesystem::path> collect_rec_input(const std::filesystem::path& path)
// {
// 	u32                                                 n_unknown_extention_files{};
// 	std::vector<std::filesystem::path>                  input_files;
// 	const std::filesystem::recursive_directory_iterator rec_dir_iter(path);
//
// 	for (const std::filesystem::path& path : rec_dir_iter) {
// 		if (std::filesystem::is_regular_file(path) && path.extension() == ".smtx") {
// 			input_files.push_back(path);
// 		} else {
// 			n_unknown_extention_files++;
// 		}
// 	}
// 	std::cout << std::format("Found in directory '{}':\n", path.string())
// 			  << std::format("\t- {} '.smtx' file(s)\n", input_files.size())
// 			  << std::format("\t- {} unsupported file(s)\n", n_unknown_extention_files);
//
// 	return input_files;
// }
//
// /*
//  * a(m, k)
//  * b(k, n)
//  * c(m, n)
//  * Expects b to be in column-major
//  */
// [[maybe_unused]] static std::vector<f32> host_spmm_rm_cm(const std::vector<f32>& a, const std::vector<f32>& b, size_t m, size_t k, size_t n)
// {
// 	std::vector<f32> res;
// 	res.reserve(m * n);
//
// 	for (size_t a_row = 0; a_row < m; ++a_row) {
// 		for (size_t b_col = 0; b_col < n; ++b_col) {
// 			f32 acc = 0;
// 			for (size_t i = 0; i < k; ++i) {
// 				acc += a[a_row * k + i] * b[b_col * k + i];
// 			}
// 			res.push_back(acc);
// 		}
// 	}
//
// 	return res;
// }
//
// [[maybe_unused]] static std::vector<f32> host_spmm_rm_rm(std::vector<f32> a, std::vector<f32> b, size_t m, size_t k, size_t n)
// {
// 	std::vector<f32> res;
// 	res.reserve(m * n);
//
// 	for (size_t a_row = 0; a_row < m; ++a_row) {
// 		for (size_t b_col = 0; b_col < n; ++b_col) {
// 			f32 acc = 0;
// 			for (size_t i = 0; i < k; ++i) {
// 				acc += a[a_row * k + i] * b[i * k + b_col];
// 			}
// 			res.push_back(acc);
// 		}
// 	}
//
// 	return res;
// }
