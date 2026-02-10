#pragma once

#include <stdio.h>

// TODO: This *can* leak memory
void* cuda_malloc_device(size_t b_size);
void* cuda_malloc_host(size_t b_size);
void  cuda_dealloc_host(void* ptr);
void  cuda_dealloc_device(void* ptr);
