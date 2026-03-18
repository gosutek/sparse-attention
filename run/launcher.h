#pragma once

#include <cstdlib>
#include <filesystem>

#include "cusparse.h"
#include "helpers.h"
#include "utils.h"

#include "spmm.h"

struct SpmmContext
{
	CSR          h_csr;
	SpMatDescr_t d_csr;

	std::vector<f32> h_dn;
	DnMatDescr_t     d_dn;

	std::vector<f32> h_res;
	DnMatDescr_t     d_res;
};

struct CusparseContext
{
	cusparseSpMatDescr_t d_csr;

	cusparseDnMatDescr_t d_dn;

	std::vector<f32>     h_res;
	cusparseDnMatDescr_t d_res;

	u64   buffer_size;
	void* buffer;

	f32 alpha, beta;
};

SpmmContext     setup_spmm(const ExecutionContext_t handle, const std::filesystem::path& sp_path);
CusparseContext setup_cusparse(const cusparseHandle_t handle, const SpMatDescr_t d_sp, const DnMatDescr_t d_dn, const DnMatDescr_t d_res);
