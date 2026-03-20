#include "launcher.h"
#include <cusparse.h>
#include <library_types.h>

SpmmContext setup_spmm(const ExecutionContext_t handle, const std::filesystem::path& sp_path)
{
	CSR          h_csr = parse_csr_dlmc(sp_path);
	SpMatDescr_t d_csr = NULL;

	CHECK_SPMM(sp_csr_create(handle, &d_csr, h_csr.rows, h_csr.cols, h_csr.nnz, h_csr.row_ptr.data(), h_csr.col_idx.data(), h_csr.val.data()));

	std::vector<f32> h_dn;
	gen_synth_weights_vec(h_dn, h_csr.cols * h_csr.cols);

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

ISpmmContext setup_ispmm(const ExecutionContext_t handle, const std::filesystem::path& sp_path)
{
	CSC          h_csc = parse_csc_dlmc(sp_path);
	SpMatDescr_t d_csc = NULL;

	CHECK_SPMM(sp_csc_create(handle, &d_csc, h_csc.rows, h_csc.cols, h_csc.nnz, h_csc.col_ptr.data(), h_csc.row_idx.data(), h_csc.val.data()));

	std::vector<f32> h_dn;
	gen_synth_weights_vec(h_dn, h_csc.rows * h_csc.rows);

	DnMatDescr_t d_dn = NULL;
	CHECK_SPMM(dn_rm_create(handle, &d_dn, h_csc.rows, h_csc.rows, h_dn.data()));

	std::vector<f32> h_res(h_csc.rows * h_csc.cols, 0);

	DnMatDescr_t d_res = NULL;
	CHECK_SPMM(dn_rm_create(handle, &d_res, h_csc.rows, h_csc.cols, h_res.data()));

	return {
		.h_csc = h_csc,
		.d_csc = d_csc,
		.h_dn = h_dn,
		.d_dn = d_dn,
		.h_res = h_res,
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

CusparseContext setup_icusparse(const cusparseHandle_t handle, const SpMatDescr_t d_sp, const DnMatDescr_t d_dn, const DnMatDescr_t d_res)
{
	constexpr const f32 alpha = 1.0f;
	constexpr const f32 beta = 0.0f;

	u32  rows, cols, nnz, *col_ptr, *row_idx;
	f32* val;

	CHECK_SPMM(sp_csc_get(d_sp, &rows, &cols, &nnz, &col_ptr, &row_idx, &val));

	// TODO: Check the ptr is on the device
	cusparseSpMatDescr_t cusparse_csc = NULL;
	CHECK_CUSPARSE(cusparseCreateCsc(&cusparse_csc,
		rows, cols, nnz,
		col_ptr, row_idx, val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	CHECK_SPMM(dn_rm_get(d_dn, &rows, &cols, &val));
	// TODO: Check the ptr is on the device
	cusparseDnMatDescr_t cusparse_dn = NULL;
	CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_dn,
		rows, cols, cols, val, CUDA_R_32F, CUSPARSE_ORDER_ROW));

	// TODO: Check the ptr is on the device
	CHECK_SPMM(dn_rm_get(d_res, &rows, &cols, &val));
	std::vector<f32>     h_res(rows * cols, 0);
	cusparseDnMatDescr_t cusparse_res = NULL;
	CHECK_CUSPARSE(cusparseCreateDnMat(&cusparse_res, rows, cols, cols, val, CUDA_R_32F, CUSPARSE_ORDER_COL));

	u64 buffer_size;
	CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&alpha, cusparse_csc, cusparse_dn, &beta, cusparse_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buffer_size));

	void* cusparse_buffer = nullptr;
	CHECK_CUDA(cudaMalloc(&cusparse_buffer, buffer_size));
	CHECK_CUSPARSE(cusparseSpMM_preprocess(handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&alpha, cusparse_csc, cusparse_dn, &beta, cusparse_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse_buffer));

	return {
		.d_csr = cusparse_csc,
		.d_dn = cusparse_dn,
		.h_res = std::move(h_res),
		.d_res = cusparse_res,
		.buffer_size = buffer_size,
		.buffer = cusparse_buffer,
		.alpha = alpha,
		.beta = beta
	};
}
