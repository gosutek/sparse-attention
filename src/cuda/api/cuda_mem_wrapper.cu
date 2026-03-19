#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	// TODO: Have these return SpmmInternal_t if the pointers are not on the device
	SpmmInternalStatus_t cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize)
	{
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyHostToDevice));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize)
	{
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyDeviceToHost));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t cuda_memset(void* s, i32 c, u64 n)
	{
		CHECK_CUDA(cudaMemset(s, c, n));
		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

#if defined(__cplusplus)
}
#endif
