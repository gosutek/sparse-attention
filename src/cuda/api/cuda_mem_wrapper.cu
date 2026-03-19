#include <cstdio>

#include "cuda_helpers.cuh"

#if defined(__cplusplus)
extern "C"
{
#endif

	// TODO: Have these return SpmmInternal_t if the pointers are not on the device
	SpmmInternalStatus_t cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize)
	{
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyHostToDevice));
	}

	SpmmInternalStatus_t cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize)
	{
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyDeviceToHost));
	}

	SpmmInternalStatus_t cuda_memset(void* s, i32 c, u64 n)
	{
		if (!_dev_ptr_chk(s))
			CHECK_CUDA(cudaMemset(s, c, n));
	}

#if defined(__cplusplus)
}
#endif
