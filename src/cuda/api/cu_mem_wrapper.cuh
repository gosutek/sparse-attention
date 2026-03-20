#pragma once

#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	SpmmInternalStatus_t cu_malloc(void* dev_ptr, const u64 bsize);
	SpmmInternalStatus_t cu_free(void* dev_ptr);
	SpmmInternalStatus_t cu_memcpy_htd(void* dst, const void* src, const u64 bsize);
	SpmmInternalStatus_t cu_memcpy_dth(void* dst, const void* src, const u64 bsize);
	SpmmInternalStatus_t cu_memset(void* s, i32 c, u64 n);

#if defined(__cplusplus)
}
#endif
