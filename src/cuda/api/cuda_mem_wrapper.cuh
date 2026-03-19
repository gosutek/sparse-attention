#pragma once

#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	SpmmInternalStatus_t cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize);
	SpmmInternalStatus_t cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize);
	SpmmInternalStatus_t cuda_memset(void* s, i32 c, u64 n);

#if defined(__cplusplus)
}
#endif
