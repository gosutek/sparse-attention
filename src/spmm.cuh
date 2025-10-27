#pragma once

#include "handle.h"
#include "matrix.h"
#include <cusparse.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// TODO: This *can* leak memory
#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUSPARSE_CHECK(x)                                                                                    \
	do {                                                                                                     \
		cusparseStatus_t err = x;                                                                            \
		if (err != CUSPARSE_STATUS_SUCCESS) {                                                                \
			fprintf(stderr, "CUSPARSE error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cusparseGetErrorString(err), cusparseGetErrorName(err), err);                                \
			abort();                                                                                         \
		}                                                                                                    \
	} while (0)

void* cuda_malloc_device(size_t b_size);
void* cuda_malloc_host(size_t b_size);
void  cuda_dealloc_host(void* ptr);
void  cuda_dealloc_device(void* ptr);

void prepare_spmm_csr(SPMM<CSR>& spmm);
void prepare_spmm_csc(SPMM<CSC>& spmm);

bool warmup_spmm_csr(SPMM<CSR>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSR>&, const uint32_t));
bool warmup_spmm_csc(SPMM<CSC>& spmm, const uint32_t size_idx, void (*run_kernel)(SPMM<CSC>&, const uint32_t));

void run_spmm_naive_elemwise_csc_gmem(SPMM<CSC>& spmm, const uint32_t idx);
void run_spmm_naive_elemwise_csc_smem(SPMM<CSC>& spmm, const uint32_t idx);
void run_spmm_coalesced_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx);
void run_spmm_blocktiling_elemwise_csr(SPMM<CSR>& spmm, const uint32_t idx);
void run_spmm_coalesced_nnzwise(SPMM<CSC>& spmm, const uint32_t idx);
void run_spmm_coalesced_nnzwise_no_smem(SPMM<CSC>& spmm, const uint32_t idx);
void run_spmm_vectorized_nnzwise_regs(SPMM<CSC>& spmm, const uint32_t idx);

void prepare_cusparse_csr(SPMM<CSR>& spmm, CuSparse& cusparse);
void prepare_cusparse_csc(SPMM<CSC>& spmm, CuSparse& cusparse);

void prepare_mhsa(MHSA<CSC, CSR>& mhsa);
void run_mhsa(MHSA<CSC, CSR>& mhsa);
