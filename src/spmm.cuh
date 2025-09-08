#pragma once

#include "handle.h"
#include "matrix.h"

void prepare_spmm(SPMM<CSC>& spmm);
void warmup_spmm(SPMM<CSC>& spmm);
void run_spmm(SPMM<CSC>& spmm);

void prepare_mhsa(MHSA<CSC, CSR>& mhsa);
void run_mhsa(MHSA<CSC, CSR>& mhsa);
