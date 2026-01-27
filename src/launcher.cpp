#include "launcher.h"

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
