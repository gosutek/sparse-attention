#include "common.h"
#include "matrix.h"
#include "model.h"

#ifndef MAT_SIZE
#	define MAT_SIZE 512
#endif

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

static void* cuda_malloc_device(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, b_size));
	return ptr;
}

void* cuda_device_copy(void* host, size_t b_size)
{
	void* device = nullptr;
	CUDA_CHECK(cudaMalloc(&device, b_size));
	// TODO: can this be async?
	CUDA_CHECK(cudaMemcpy(device, host, b_size, cudaMemcpyHostToDevice));
	return device;
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

__device__ inline static float get_element_rm(const float* const a, size_t n_cols, size_t row, size_t col)
{
	return a[row * n_cols + col];
}

__device__ inline static float get_element_cm(const float* const a, size_t n_rows, size_t row, size_t col)
{
	return a[col * n_rows + row];
}

__device__ inline static void set_element_rm(float* const a, size_t n_cols, size_t row, size_t col, float val)
{
	a[row * n_cols + col] = val;
}

__device__ inline static void set_element_cm(float* const a, size_t n_rows, size_t row, size_t col, float val)
{
	a[col * n_rows + row] = val;
}

__global__ void spmm_rm_csc(
	const float* __restrict__ a,  // expect row-major for coalesced access
	const uint32_t* __restrict__ col_ptr,
	const uint32_t* __restrict__ row_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* __restrict__ res)  // expect row-major for coalesced access
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= n || y >= m) {
		return;
	}

	float acc = 0.0f;
	for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
		acc += get_element_rm(a, k, y, row_idx[i]) * val[i];
	}
	set_element_rm(res, n, y, x, acc);
}

__global__ void spmm_rm_csr(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
	const float* __restrict__ val,
	const uint32_t m,
	const uint32_t k,
	const uint32_t n,
	float* const __restrict__ res)  // expect row-major for coalesced access
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= n || y >= m) {
		return;
	}

	float acc = 0.0f;
	for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
		acc += get_element_rm(a, k, y, col_idx[i]) * val[i];
	}
	set_element_rm(res, n, y, x, acc);
}

void run(CSC_MHSA mhsa)
{
	CSCMatrix& w_q = mhsa.weights.w_q[0];
	// TODO: change MAT_SIZE
	float* res = static_cast<float*>(cuda_malloc_device(sizeof(float) * mhsa.config.input_sequence_size * MAT_SIZE));
	// TODO: Merge these two ^ v allocations
	void* dev = cuda_device_copy(mhsa.host, mhsa.b_size);

	float*    x = reinterpret_cast<float*>(dev);
	uint32_t* d_col_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(dev) + (mhsa.config.input_sequence_size * MAT_SIZE) * sizeof(float));
	uint32_t* d_row_idx = d_col_ptr + w_q.col_ptr_size;
	float*    d_val = reinterpret_cast<float*>(d_row_idx + w_q.row_idx_size);

	const uint32_t M = mhsa.config.input_sequence_size;
	const uint32_t K = w_q.rows;
	const uint32_t N = w_q.cols;

	dim3 dimBlock(32, 32);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
		(M + dimBlock.y - 1) / dimBlock.y);

#if defined(__CHRONO__)
	cudaEvent_t start, stop;
	float       time;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start, 0));
#endif

	spmm_rm_csc<<<dimGrid, dimBlock>>>(x, d_col_ptr, d_row_idx, d_val, M, K, N, res);

#if defined(__CHRONO__)
	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));

	CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	std::cout << "Clock: " << time << "ms" << std::endl;
#endif

	CUDA_CHECK(cudaDeviceSynchronize());

	// TODO: can this be async?
	CUDA_CHECK(cudaMemcpy(mhsa.host, res, sizeof(float) * mhsa.config.input_sequence_size * MAT_SIZE, cudaMemcpyDeviceToHost));

	cuda_dealloc_device(res);
	cuda_dealloc_device(dev);
}
