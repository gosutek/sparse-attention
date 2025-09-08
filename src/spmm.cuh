#pragma once

#include "handle.h"
#include "matrix.h"

void* cuda_malloc_device(size_t b_size);
void* cuda_malloc_host(size_t b_size);
void  cuda_dealloc_host(void* ptr);
void  cuda_dealloc_device(void* ptr);

void prepare_spmm(SPMM<CSC>& spmm);
void warmup_spmm(SPMM<CSC>& spmm, const uint8_t size_idx);
void run_spmm(SPMM<CSC>& spmm, const uint8_t idx);

void prepare_mhsa(MHSA<CSC, CSR>& mhsa);
void run_mhsa(MHSA<CSC, CSR>& mhsa);
