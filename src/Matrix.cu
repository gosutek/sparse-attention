#include <cstdio>
// clang-format off
#include <stdio.h>
#include "../include/mmio.h"
// clang-format on
#include <stdexcept>

class BCSRMatrix
{
private:
	// TODO: Do I need these 3?
	uint32_t rows = 0;
	uint32_t cols = 0;
	uint32_t nnz = 0;

	uint32_t m_val_byte_size = 0;
	uint32_t m_row_ptr_byte_size = 0;
	uint32_t m_col_idx_byte_size = 0;
	uint32_t m_nzptr_byte_size = 0;

	// TODO: This should be fp8
	uint32_t* val = nullptr;
	uint32_t* col_idx = nullptr;
	uint32_t* row_ptr = nullptr;
	uint32_t* nzptr = nullptr;

public:
};
