#pragma once

#include "matrix.h"

struct Weights
{
	CSRMatrix x;  // (vocab, d_m)

	CSRMatrix w_q;
	CSRMatrix w_k;
	CSRMatrix w_v;
	CSRMatrix w_o;
};

struct MHSA
{
	Weights weights;
	size_t  b_size;
};
