#pragma once

#include "helpers.h"
#include <stdio.h>

#if defined(__cplusplus)
extern "C"
{
#endif

	void cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize);
	void cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize);

#if defined(__cplusplus)
}
#endif
