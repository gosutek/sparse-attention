#pragma once

#include "helpers.h"
#include <stdio.h>

void cuda_mem_cpy_hd(void* dst, const void* src, const u64 bsize);
void cuda_mem_cpy_dh(void* dst, const void* src, const u64 bsize);
