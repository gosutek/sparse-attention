#include "../cu_helpers.cuh"
#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	SpmmInternalStatus_t cu_malloc(void* dev_ptr, const u64 bsize)
	{
		if (!_dev_ptr_chk(dev_ptr)) {
			return SPMM_INTERNAL_STATUS_POINTER_INVALID_MEM_TYPE;
		}
		CHECK_CUDA(cudaMalloc(&dev_ptr, bsize));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cu_free(void* dev_ptr)
	{
		if (!_dev_ptr_chk(dev_ptr)) {
			return SPMM_INTERNAL_STATUS_POINTER_INVALID_MEM_TYPE;
		}
		CHECK_CUDA(cudaFree(dev_ptr));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cu_memcpy_htd(void* dst, const void* src, const u64 bsize)
	{
		if (!_dev_ptr_chk(dst)) {
			return SPMM_INTERNAL_STATUS_POINTER_INVALID_MEM_TYPE;
		}
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyHostToDevice));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cu_memcpy_dth(void* dst, const void* src, const u64 bsize)
	{
		if (!_dev_ptr_chk(src)) {
			return SPMM_INTERNAL_STATUS_POINTER_INVALID_MEM_TYPE;
		}
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyDeviceToHost));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cu_memset(void* s, i32 c, u64 n)
	{
		if (!_dev_ptr_chk(s)) {
			return SPMM_INTERNAL_STATUS_POINTER_INVALID_MEM_TYPE;
		}
		CHECK_CUDA(cudaMemset(s, c, n));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

#if defined(__cplusplus)
}
#endif
