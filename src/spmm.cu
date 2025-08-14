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

/*
 * Column-Major access pattern
 * x * n_rows + y
 */
__global__ void spmm_rm_csc(
	const float* const    a,  // expect col-major for coalesced access
	const uint32_t* const col_ptr,
	const uint32_t* const row_idx,
	const float* const    val,
	const uint32_t        n_rows,
	const uint32_t        n_cols,
	float* const          res)  // expect row-major for coalesced access
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < n_rows) {
		float acc = 0;
		for (size_t i = col_ptr[x]; i < col_ptr[x + 1]; ++i) {
			// printf("Multiplying %f x %f\n", val[i], a[y * n_rows + row_idx[i]]);
			acc += val[i] * a[y * n_rows + row_idx[i]];
		}
		res[y * n_cols + x] = acc;
	}
}

/*
 * Row-Major access pattern
 * y * n_cols + x
 */
__global__ void spmm_rm_csr(
	const float* const    a,
	const uint32_t* const row_ptr,
	const uint32_t* const col_idx,
	const float* const    val,
	const uint32_t        n_rows,
	const uint32_t        n_cols,
	float* const          res)  // expect row-major for coalesced access
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < n_rows) {
		float acc = 0;
		for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
			acc += val[i] * a[y * n_cols + col_idx[i]];
		}
		res[y * n_cols + x] = acc;
	}
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

	dim3 dimBlock(16, 16);
	dim3 dimGrid(32, 32);

#if defined(__CHRONO__)
	cudaEvent_t start, stop;
	float       time;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start, 0));
#endif

	spmm_rm_csc<<<dimGrid, dimBlock>>>(x, d_col_ptr, d_row_idx, d_val, w_q.rows, w_q.cols, res);

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
