#pragma once

#include "matrix.h"

void run_spmm(std::filesystem::path& path);
void run_mhsa(MHSA<CSC, CSR>& mhsa);
