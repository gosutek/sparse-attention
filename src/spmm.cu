#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cusparse.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "handle.h"
#include "matrix.h"
#include "spmm.cuh"
#include "utils.h"

enum class KernelType
{
	GlobalMemory,
	SharedMemory,
};

enum class OutputFormat
{
	RM,
	CM
};

void* cuda_malloc_device(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, b_size));
	return ptr;
}

void* cuda_malloc_host(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMallocHost(&ptr, b_size));
	return ptr;
}

void cuda_dealloc_host(void* ptr)
{
	CUDA_CHECK(cudaFreeHost(ptr));
}

void cuda_dealloc_device(void* ptr)
{
	CUDA_CHECK(cudaFree(ptr));
}

__device__ inline static float get_elem_rm(const float* const a, size_t n_cols, size_t row, size_t col)
{
	return a[row * n_cols + col];
}

[[maybe_unused]] __device__ inline static float get_elem_cm(const float* const a, size_t n_rows, size_t row, size_t col)
{
	return a[col * n_rows + row];
}

__device__ inline static void set_elem_rm(float* const a, size_t n_cols, size_t row, size_t col, float val)
{
	a[row * n_cols + col] = val;
}

__device__ inline static void set_elem_cm(float* const a, size_t n_rows, size_t row, size_t col, float val)
{
	a[col * n_rows + row] = val;
}

template <KernelType K, OutputFormat O>
__global__ void spmm_csc(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x, y;
	if constexpr (K == KernelType::SharedMemory) {
		x = threadIdx.x;
		y = blockIdx.x;
	} else {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		y = blockIdx.y * blockDim.y + threadIdx.y;
	}

	float acc = 0.0f;
	if constexpr (K == KernelType::SharedMemory) {
		__shared__ float x_row_sm[MAT_SIZE];

		x_row_sm[x] = get_elem_rm(a, k, y, x);

		__syncthreads();
		for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
			acc += x_row_sm[row_idx[i]] * val[i];
		}
	} else {
		for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
			acc += get_elem_rm(a, k, y, row_idx[i]) * val[i];
		}
	}
	if constexpr (O == OutputFormat::RM) {
		set_elem_rm(res, n, y, x, acc);
	} else {
		set_elem_cm(res, m, y, x, acc);
	}
}

__global__ void spmm_coalesced_csr(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = threadIdx.x;
	uint32_t y = blockIdx.x;

	__shared__ float x_row_sm[MAT_SIZE];
	__shared__ float shared_acc[MAT_SIZE];

	for (uint32_t i = x; i < k; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, y, i);
		shared_acc[i] = 0.0f;
	}
	__syncthreads();

	for (uint32_t row = 0; row < k; ++row) {
		for (uint32_t i = row_ptr[row] + x; i < row_ptr[row + 1]; i += blockDim.x) {
			// TODO: Figure out a way to remove atomicAdd
			atomicAdd_block(&shared_acc[col_idx[i]], x_row_sm[row] * val[i]);
		}
	}

	__syncthreads();

	for (uint32_t i = x; i < k; i += blockDim.x) {
		set_elem_rm(res, n, y, i, shared_acc[i]);
	}
}

__global__ void spmm_csc_1d_blocktiling(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	__shared__ float x_row_sm[MAT_SIZE];

	for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, blockIdx.y, i);
	}
	__syncthreads();

	float acc = 0.0f;
	for (size_t i = col_ptr[blockIdx.x] + threadIdx.x; i < col_ptr[blockIdx.x + 1]; i += blockDim.x) {
		acc += x_row_sm[row_idx[i]] * val[i];
	}
	__syncthreads();

	for (uint8_t i = WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, WARP_SIZE);
	}

	uint32_t lane_id = threadIdx.x & 0x1f;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	constexpr uint8_t n_warps = N_THREADS / WARP_SIZE;
	__shared__ float  warp_sums[n_warps];

	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}

	__syncthreads();

	if (warp_id == 0) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane_id];

		constexpr uint32_t mask = 0xFF;

		for (uint8_t i = n_warps / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, WARP_SIZE);
		}

		if (lane_id == 0) {
			set_elem_rm(res, n, blockIdx.y, blockIdx.x, acc);
		}
	}
}

__global__ void spmm_csc_2d_blocktiling(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	__shared__ float x_row_sm[MAT_SIZE];

	// for (size_t i = blockIdx.z * BK + threadIdx.x * TK; i < (blockIdx.z + 1) * BK; i += blockDim.x * TK) {
	for (size_t i = threadIdx.x * TK; i < MAT_SIZE; i += blockDim.x * TK) {
		float4 tmp = reinterpret_cast<const float4*>(&a[blockIdx.y * k + i])[0];
#pragma unroll
		for (uint8_t t = 0; t < TK; t++) {
			x_row_sm[i + t] = ((float*)&tmp)[t];
		}
	}
	__syncthreads();

	float  acc = 0.0f;
	size_t nz_start = col_ptr[blockIdx.x];
	size_t nz_end = col_ptr[blockIdx.x + 1];
	size_t nnz_per_block = (nz_end - nz_start + gridDim.z - 1) / gridDim.z;

	size_t block_start = nz_start + blockIdx.z * nnz_per_block;
	size_t block_end = min(block_start + nnz_per_block, nz_end);

	for (size_t i = block_start + threadIdx.x * TK;
		i < block_end;
		i += blockDim.x * TK) {
		size_t thread_end = min(i + TK, block_end);
		// TODO: Make into vectorized loads
#pragma unroll
		for (size_t j = i; j < thread_end; ++j) {
			acc += x_row_sm[row_idx[j]] * val[j];
		}
	}
	__syncthreads();

	for (uint8_t i = WARP_SIZE / 2; i > 0; i /= 2) {
		acc += __shfl_xor_sync(0xffffffff, acc, i, WARP_SIZE);
	}

	uint32_t lane_id = threadIdx.x & 0x1f;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	constexpr uint8_t n_warps = N_THREADS / WARP_SIZE;
	__shared__ float  warp_sums[n_warps];

	// at this point the first thread (lane_id == 0) of every warp in this block
	// has the result from TN non-zeros for this col
	// this is essentially warp-wide reduction
	if (lane_id == 0) {
		warp_sums[warp_id] = acc;
	}
	// we write the warp-wide results to a block-wide memory location (SMEM)
	// so that we can perform block-wide reduction

	__syncthreads();

	// we assign the block-wide reduction to warp-0
	if (warp_id == 0) {
		// WARN: some threads point to garbage
		float acc = warp_sums[lane_id];

		constexpr uint32_t mask = 0x3;

		for (uint8_t i = n_warps / 2; i > 0; i /= 2) {
			acc += __shfl_xor_sync(mask, acc, i, n_warps);
		}
		if (lane_id == 0) {
			atomicAdd(&res[blockIdx.y * n + blockIdx.x], acc);
		}
	}
}

__global__ void spmm_1d_blocktiling(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	__shared__ float x_row_sm[MAT_SIZE];
	__shared__ float shared_acc[MAT_SIZE];

	// Does indeed load the whole of row from A for every block (k / BN times)
	for (size_t i = threadIdx.x; i < k; i += blockDim.x) {
		x_row_sm[i] = get_elem_rm(a, k, blockIdx.x, i);
		shared_acc[i] = 0.0f;
	}
	__syncthreads();

	for (size_t r = 0; r < k; ++r) {
		size_t bound = min(static_cast<unsigned long>(row_ptr[r + 1]), row_ptr[r] + (blockIdx.y + 1) * BN);
		for (size_t i = row_ptr[r] + blockIdx.y * BN + threadIdx.x; i < bound; i += blockDim.x) {
			atomicAdd(&shared_acc[col_idx[i]], val[i] * x_row_sm[r]);
			// if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
			// 	printf("Multiplying val[%lu](%.2f) * x_row_sm[%lu](%.2f) and storing at shared_acc[%lu](%.2f)\n",
			// 		i, val[i], r, x_row_sm[r], col_idx[i], shared_acc[col_idx[i]]);
			// }
		}
	}

	__syncthreads();

	for (size_t i = threadIdx.x; i < MAT_SIZE; i += blockDim.x) {
		if (shared_acc[i] != 0) {
			set_elem_rm(res, n, blockIdx.x, i, shared_acc[i]);
		}
	}
}

template <OutputFormat O>
__global__ void spmm_csc_memio(
	const float* __restrict__ a,
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t rect = threadIdx.x;
	uint32_t y = blockIdx.x;

	float            acc[TN] = { 0.0f };
	__shared__ float x_row_sm[MAT_SIZE];

	for (size_t x = rect * TN; x < rect * TN + TN; ++x) {
		x_row_sm[x] = get_elem_rm(a, k, y, x);
	}
	__syncthreads();

	for (size_t x = rect * TN; x < rect * TN + TN; ++x) {
		size_t idx = x % TN;
		for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
			acc[idx] += x_row_sm[row_idx[i]] * val[i];
		}
		if constexpr (O == OutputFormat::RM) {
			set_elem_rm(res, n, y, x, acc[idx]);
		} else {
			set_elem_cm(res, m, y, x, acc[idx]);
		}
	}
}

// TODO: Incorporate into the template
__global__ void spmm_rm_csr_gm(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= n || y >= m) {
		return;
	}

	float acc = 0.0f;
	for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
		acc += get_elem_rm(a, k, y, col_idx[i]) * val[i];
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void gemm(
	const float* __restrict__ a,  // row-major
	const float* __restrict__ b,  // col-major
	const size_t m,
	const size_t k,
	const size_t n,
	float* __restrict__ res)
{
	uint32_t x = threadIdx.x;
	uint32_t y = blockIdx.x;

	if (x >= n || y >= m) {  // not really needed
		return;
	}

	float acc = 0.0f;
	// TODO: Change hardcoded value
	__shared__ float a_row_sm[512];

	a_row_sm[x] = get_elem_rm(a, k, y, x);
	__syncthreads();

	for (size_t i = 0; i < k; ++i) {
		acc += a_row_sm[i] * b[x * k + i];
	}
	set_elem_rm(res, n, y, x, acc);
}

__global__ void softmax(
	const float* __restrict__ a,
	const size_t m,
	const size_t k,
	float*       acc,
	float* __restrict__ res)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	// TODO: std::expf()
	float e = std::exp(get_elem_rm(a, k, y, x));
	atomicAdd(acc, e);

	__syncthreads();

	float val = e / *acc;
	set_elem_rm(res, k, y, x, val);
}

void prepare_cusparse_csr(SPMM<CSR>& spmm, CuSparse& cusparse)
{
	CUSPARSE_CHECK(cusparseCreateCsr(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

void prepare_cusparse_csc(SPMM<CSC>& spmm, CuSparse& cusparse)
{
	CUSPARSE_CHECK(cusparseCreateCsc(&cusparse.sparse,
		spmm.dev.s.rows, spmm.dev.s.cols, spmm.host.s.nnz,
		spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	size_t tmp = 0;
	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.dense[i], BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.rows, spmm.dev.s.rows, spmm.dev.d[i], CUDA_R_32F, CUSPARSE_ORDER_ROW));
		CUSPARSE_CHECK(cusparseCreateDnMat(&cusparse.res[i], spmm.dev.s.cols, BENCHMARKING_DENSE_N_ROWS[i], spmm.dev.s.cols, spmm.dev.r[i], CUDA_R_32F, CUSPARSE_ORDER_COL));

		CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &tmp));

		cusparse.work_buffer_size += tmp;
	}

	cusparse.work_buffer = cuda_malloc_device(cusparse.work_buffer_size);
	if (!cusparse.work_buffer) {
		throw std::runtime_error("Failed to allocate work buffer of size: " + std::to_string(cusparse.work_buffer_size));
	}
}

void prepare_spmm_csr(SPMM<CSR>& spmm)
{
	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
	}

	std::ifstream file_stream = { spmm.sparse_path };
	DLMCHeader    header = parse_dlmc_header(file_stream);
	size_t        sparse_b_size = (sizeof(uint32_t) * (header.n_rows + 1) + sizeof(uint32_t) * header.nnz + sizeof(float) * header.nnz);

	/**
    * Twice the total size of the dense matrices.
    * Once for the input
    * Twice for the result
    **/
	spmm.host.data = cuda_malloc_host(sparse_b_size + 2 * BENCHMARKING_TOTAL_DENSE_B_SIZE);
	spmm.host.d[0] = reinterpret_cast<float*>(spmm.host.data);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCHMARKING_DENSE_N_ROWS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
		}
	}

	void* start_of_sparse = spmm.host.d[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] +                          // from the last ptr of spmm.host.d
	                        BENCHMARKING_DENSE_N_ROWS[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] * MAT_SIZE;  // skip 512 * 512 floats
	spmm.host.s = parse_csr_dlmc(start_of_sparse, spmm.sparse_path);

	float* ptr = spmm.host.s.val + spmm.host.s.val_size;

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.host.r[i] = ptr;
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
	}

	/*
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
      * | x_32 | x_64 | x_128 | x_256 | x_512 | row_ptr | col_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
      * +------------------------------------------HOST/DEVICE------------------------------------------------+
   */

	spmm.dev.data = cuda_malloc_device(spmm.host.s.b_size + 2 * BENCHMARKING_TOTAL_DENSE_B_SIZE);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCHMARKING_TOTAL_DENSE_B_SIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<float*>(spmm.dev.data);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.d[i] = ptr;
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
	}

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSR(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr = spmm.dev.s.val + spmm.dev.s.val_size;

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.r[i] = ptr;
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
	}
}

void prepare_spmm_csc(SPMM<CSC>& spmm)
{
	if (!std::filesystem::exists(spmm.sparse_path) || !std::filesystem::is_regular_file(spmm.sparse_path)) {
		throw std::runtime_error("Invalid file given: " + spmm.sparse_path.string());
	}

	std::ifstream file_stream = { spmm.sparse_path };
	DLMCHeader    header = parse_dlmc_header(file_stream);

	size_t col_ptr_b_size = sizeof(uint32_t) * (header.n_cols + 1);
	size_t row_idx_b_size = sizeof(uint32_t) * header.nnz;
	size_t val_b_size = sizeof(float) * header.nnz;
	size_t sparse_b_size_aligned = col_ptr_b_size + calc_padding_bytes(col_ptr_b_size, ALIGNMENT_BYTES) +
	                               row_idx_b_size + calc_padding_bytes(row_idx_b_size, ALIGNMENT_BYTES) +
	                               val_b_size + calc_padding_bytes(val_b_size, ALIGNMENT_BYTES);

	/**
    * Twice the total size of the dense matrices.
    * Once for the input
    * Twice for the result
    **/
	spmm.b_size = sparse_b_size_aligned + 2 * BENCHMARKING_TOTAL_DENSE_B_SIZE;
	spmm.host.data = cuda_malloc_host(spmm.b_size);
	spmm.host.d[0] = reinterpret_cast<float*>(spmm.host.data);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		generate_token_embeddings(spmm.host.d[i], BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE);
		if (i + 1 < std::size(BENCHMARKING_DENSE_N_ROWS)) {
			spmm.host.d[i + 1] = spmm.host.d[i] + BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE;
		}
	}

	assert((reinterpret_cast<uintptr_t>(spmm.host.d[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.d[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	void* start_of_sparse = spmm.host.d[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] +                          // from the last ptr of spmm.host.d
	                        BENCHMARKING_DENSE_N_ROWS[std::size(BENCHMARKING_DENSE_N_ROWS) - 1] * MAT_SIZE;  // skip 512 * 512 floats

	// start_of_sparse is 128-byte aligned guaranteed
	spmm.host.s = parse_csc_dlmc(start_of_sparse, spmm.sparse_path);

	assert((reinterpret_cast<uintptr_t>(spmm.host.s.col_ptr) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.row_idx) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.s.val) & (ALIGNMENT_BYTES - 1)) == 0);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(start_of_sparse) + spmm.host.s.b_size;

	// TODO: use uintptr_t instead of pointer arithmetic on float* (??)
	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.host.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[0]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[1]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[2]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[3]) & (ALIGNMENT_BYTES - 1)) == 0);
	assert((reinterpret_cast<uintptr_t>(spmm.host.r[4]) & (ALIGNMENT_BYTES - 1)) == 0);

	// WARN: asserts cost
	assert(sparse_b_size_aligned == spmm.host.s.b_size);

	/*
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-------+-------+-------+
      * | x_32 | x_64 | x_128 | x_256 | x_512 | col_ptr | row_idx | val | r_32 | r_64 | r_128 | r_256 | r_512 |
      * +------+------+-------+-------+-------+---------+---------+-----+------+------+-----+---+-------------+
      * +------------------------------------------HOST/DEVICE------------------------------------------------+
   */

	spmm.dev.data = cuda_malloc_device(spmm.b_size);
	CUDA_CHECK(cudaMemcpy(spmm.dev.data, spmm.host.data, spmm.host.s.b_size + BENCHMARKING_TOTAL_DENSE_B_SIZE, cudaMemcpyHostToDevice));

	// Partition dev
	ptr = reinterpret_cast<uintptr_t>(spmm.dev.data);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.d[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}

	// TODO: This trashes the previous empty object and makes a new one. Make a good copy assignment operator function instead.
	spmm.dev.s = CSC(spmm.host.s.rows, spmm.host.s.cols, spmm.host.s.nnz);
	spmm.dev.s.partition(ptr);

	ptr += spmm.host.s.b_size;

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		spmm.dev.r[i] = reinterpret_cast<float*>(ptr);
		ptr += BENCHMARKING_DENSE_N_ROWS[i] * MAT_SIZE * sizeof(float);
	}
}

void warmup_spmm_csr(SPMM<CSR>& spmm, const uint8_t size_idx)
{
	// PERF: Bounds check
	assert(size_idx < std::size(BENCHMARKING_DENSE_N_ROWS) - 1);
	run_spmm_csr(spmm, size_idx);

	const size_t res_size = BENCHMARKING_DENSE_N_ROWS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csr(spmm, cusparse);

	CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, cusparse.work_buffer));
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
}

void warmup_spmm_csc(SPMM<CSC>& spmm, const uint8_t size_idx)
{
	// PERF: Bounds check
	assert(size_idx < std::size(BENCHMARKING_DENSE_N_ROWS) - 1);
	run_spmm_csc(spmm, size_idx);

	const size_t res_size = BENCHMARKING_DENSE_N_ROWS[size_idx] * MAT_SIZE;
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	// WARN: Temporary hack
	std::memcpy(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size * sizeof(float));

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);
	prepare_cusparse_csc(spmm, cusparse);

	CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&cusparse.alpha, cusparse.sparse, cusparse.dense[size_idx], &cusparse.beta, cusparse.res[size_idx], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, cusparse.work_buffer));
	CUDA_CHECK(cudaMemcpy(spmm.host.r[size_idx], spmm.dev.r[size_idx], res_size * sizeof(float), cudaMemcpyDeviceToHost));

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	verify_res(spmm.host.r[size_idx + 1], spmm.host.r[size_idx], res_size);
}

void run_spmm_csr(SPMM<CSR>& spmm, const uint8_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	// One thread per element of the output.
	// One thread block stretched across a row of the output
	// (32x512)*(512x512)=(32x512)
	// dim3 spmm_block_sm(MAT_SIZE / TN);
	// dim3 spmm_grid_sm(BENCHMARKING_DENSE_N_ROWS[idx]);
	// dim3 spmm_block_sm(256);
	// dim3 spmm_grid_sm(BENCHMARKING_DENSE_N_ROWS[idx]);
	dim3 spmm_1d_blocktiling_grid_size(BENCHMARKING_DENSE_N_ROWS[idx], n / BN);
	dim3 spmm_1d_blocktiling_block_size(BN / TN);

	spmm_1d_blocktiling<<<spmm_1d_blocktiling_grid_size, spmm_1d_blocktiling_block_size>>>(
		spmm.dev.d[idx],
		spmm.dev.s.row_ptr, spmm.dev.s.col_idx, spmm.dev.s.val,
		m, k, n, spmm.dev.r[idx]);
}

void run_spmm_csc(SPMM<CSC>& spmm, const uint8_t idx)
{
	const size_t m = BENCHMARKING_DENSE_N_ROWS[idx];
	const size_t k = spmm.dev.s.rows;
	const size_t n = spmm.dev.s.cols;

	// NOTE: 1d_blocktiling
	dim3 spmm_grid_sm(n, BENCHMARKING_DENSE_N_ROWS[idx]);
	dim3 spmm_block_sm(N_THREADS);
	spmm_csc_1d_blocktiling<<<spmm_grid_sm, spmm_block_sm>>>(
		spmm.dev.d[idx],
		spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
		m, k, n, spmm.dev.r[idx]);

	// NOTE: 2d_blocktiling
	// // PERF: Hack ~ find a better way to deal with having to add instead of set the result
	// const size_t res_size = BENCHMARKING_DENSE_N_ROWS[idx] * MAT_SIZE * sizeof(float);
	// CUDA_CHECK(cudaMemset(spmm.dev.r[idx], 0, res_size));
	// CUDA_CHECK(cudaDeviceSynchronize());
	// dim3 grid(n, BENCHMARKING_DENSE_N_ROWS[idx], k / BK);
	// dim3 block(N_THREADS);
	// spmm_csc_2d_blocktiling<<<grid, block>>>(
	// 	spmm.dev.d[idx],
	// 	spmm.dev.s.col_ptr, spmm.dev.s.row_idx, spmm.dev.s.val,
	// 	m, k, n, spmm.dev.r[idx]);
}

// void prepare_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	// mhsa_load_host_csc(mhsa, mhsa.config, mhsa.dlmc, mhsa.weights);
//
// 	// TODO: Find a better name
// 	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
// 	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;
//
// 	size_t res_b_size = sizeof(float) * (kv_size * 4 + gemm_res_size * 2 + 1);  // Q, K, V, gemm result, float acc for softmax, Attention matrix, Final Result
//
// 	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
// 	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));
//
// 	/*
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * | x | w_q | w_k | w_v | w_o | mask | Q | K | V | QK^T | ACC | A | Final Result |
//       * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
//       * +-------------HOST-----------------+----------------DEVICE---------------------+
//    */
//
// 	res.x = reinterpret_cast<float*>(mhsa.dev);
// 	size_t b_x_size = sizeof(float) * kv_size;
//
// 	char* ptr = reinterpret_cast<char*>(res.x) + b_x_size;
//
// 	// TODO: This call copy assignment operator of CSC
// 	// check if the custom one does what you want
// 	res.w_q = mhsa.weights.w_q[0];
// 	res.w_q.partition(ptr);
// 	ptr += res.w_q.b_size;
//
// 	res.w_k = mhsa.weights.w_k[0];
// 	res.w_k.partition(ptr);
// 	ptr += res.w_k.b_size;
//
// 	res.w_v = mhsa.weights.w_v[0];
// 	res.w_v.partition(ptr);
// 	ptr += res.w_v.b_size;
//
// 	res.w_o = mhsa.weights.w_o[0];
// 	res.w_o.partition(ptr);
// 	ptr += res.w_o.b_size;
//
// 	res.q_res = reinterpret_cast<float*>(ptr);
// 	res.k_res = res.q_res + kv_size;
// 	res.v_res = res.k_res + kv_size;
// 	res.gemm_res = res.v_res + kv_size;
// 	res.softmax_acc = res.gemm_res + gemm_res_size;
// 	res.attention = res.softmax_acc + 1;
//
// 	return res;
// }
//
// void run_mhsa(MHSA<CSC, CSR>& mhsa)
// {
// 	DevMHSA      d = prepare_mhsa(mhsa);
// 	const size_t m = mhsa.config.input_sequence_size;
// 	const size_t n = d.w_q.cols;
//
// 	// One thread per element of the output
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_gm(32, 32);
// 	dim3 spmm_grid_gm(
// 		(n + spmm_block_gm.x - 1) / spmm_block_gm.x,
// 		(m + spmm_block_gm.y - 1) / spmm_block_gm.y);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x512)=(32x512)
// 	dim3 spmm_block_sm(512);
// 	dim3 spmm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block stretched across a row of the output
// 	// (32x512)*(512x32)=(32x32)
// 	dim3 gemm_block_sm(32);
// 	dim3 gemm_grid_sm(32);
//
// 	// One thread per element of the output.
// 	// One thread block per 32x32 submatrix of the output
// 	// (32x32)
// 	dim3 softmax_block(32, 32);
// 	dim3 softmax_grid(
// 		(m + softmax_block.x - 1) / softmax_block.x,
// 		(m + softmax_block.y - 1) / softmax_block.y);  // This should actually be equal to (1,1) i.e. one block
//
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_q.col_ptr, d.w_q.row_idx, d.w_q.val, mhsa.config.input_sequence_size, d.w_q.rows, d.w_q.cols, d.q_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_k.col_ptr, d.w_k.row_idx, d.w_k.val, mhsa.config.input_sequence_size, d.w_k.rows, d.w_k.cols, d.k_res);
// 	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_v.col_ptr, d.w_v.row_idx, d.w_v.val, mhsa.config.input_sequence_size, d.w_v.rows, d.w_v.cols, d.v_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	gemm<<<gemm_grid_sm, gemm_block_sm>>>(d.q_res, d.k_res, mhsa.config.input_sequence_size, d.w_q.rows, mhsa.config.input_sequence_size, d.gemm_res);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	softmax<<<softmax_grid, softmax_block>>>(d.gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, d.softmax_acc, d.attention);
//
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// TODO: can this be async?
// 	// TODO: THIS NEEDS TO WRITE TO PAGE-LOCKED MEMORY NOT SOME RANDOM ALLOCATED MEMORY
// 	//
// 	// CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
// }
