#include "cuda_mem_wrapper.cuh"

void cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize)
{
	CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyHostToDevice));
}

void cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize)
{
	CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyDeviceToHost));
}
