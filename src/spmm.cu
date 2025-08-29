#include "common.h"
#include "matrix.h"
#include "model.h"

#ifndef MAT_SIZE
#	define MAT_SIZE 512
#endif

// TODO: This *can* leak memory
#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

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

static void* cuda_malloc_device(size_t b_size)
{
	void* ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&ptr, b_size));
	return ptr;
}

// void* cuda_device_copy(void* host, size_t b_size)
// {
// 	void* device = nullptr;
// 	CUDA_CHECK(cudaMalloc(&device, b_size));
// 	// TODO: can this be async?
// 	CUDA_CHECK(cudaMemcpy(device, host, b_size, cudaMemcpyHostToDevice));
// 	return device;
// }

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

void print_device_properties()
{
	cudaDeviceProp dev_prop = {};
	CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));

	printf(
		"# CUDA: %s, compute %d.%d, %d SMs, %.1f GiB, peak bandwidth %.0f GB/s\n- Maximum threads per block ~ %d\n",
		dev_prop.name, dev_prop.major, dev_prop.minor, dev_prop.multiProcessorCount,
		static_cast<double>(dev_prop.totalGlobalMem) / (1024 * 1024 * 1024),
		static_cast<double>(dev_prop.memoryClockRate) * (dev_prop.memoryBusWidth / 8) * 2 / 1e6,
		dev_prop.maxThreadsPerBlock);
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
	const float* __restrict__ a,  // expect row-major for coalesced access
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

	if (x >= n || y >= m) {  // not really needed
		return;
	}

	float acc = 0.0f;
	if constexpr (K == KernelType::SharedMemory) {
		// TODO: Change hardcoded value
		__shared__ float x_row_sm[512];

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

// TODO: Incorporate into the template
__global__ void spmm_rm_csr_gm(
	const float* __restrict__ a,
	const uint32_t* __restrict__ row_ptr,
	const uint32_t* __restrict__ col_idx,
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

	float e = std::exp(get_elem_rm(a, k, y, x));
	atomicAdd(acc, e);

	__syncthreads();

	float val = e / *acc;
	set_elem_rm(res, k, y, x, val);
}

// TODO: Template this?
void run(MHSA<CSC, CSR>& mhsa)
{
	// TODO: Find a better name
	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;
	size_t res_b_size = sizeof(float) * kv_size * 3 + gemm_res_size + 1;  // Q, K, V, gemm result, float acc for softmax
	void*  dev = cuda_malloc_device(mhsa.b_size + res_b_size);
	CUDA_CHECK(cudaMemcpy(dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));

	/*
      * +---+-----+-----+-----+-----+---+---+---+------+-----+
      * | x | w_q | w_k | w_v | w_o | Q | K | V | QK^T | ACC |
      * +---+-----+-----+-----+-----+---+---+---+------+-----+
   */

	float* x = reinterpret_cast<float*>(dev);
	size_t b_x_size = sizeof(float) * kv_size;

	char* ptr = reinterpret_cast<char*>(x) + b_x_size;

	CSC d_wq = mhsa.weights.w_q[0];
	d_wq.partition(ptr);
	ptr += d_wq.b_size;

	CSC d_wk = mhsa.weights.w_k[0];
	d_wk.partition(ptr);
	ptr += d_wk.b_size;

	CSC d_wv = mhsa.weights.w_v[0];
	d_wv.partition(ptr);
	ptr += d_wv.b_size;

	CSC d_wo = mhsa.weights.w_o[0];
	d_wo.partition(ptr);
	ptr += d_wo.b_size;

	float* q_res = reinterpret_cast<float*>(ptr);
	float* k_res = q_res + kv_size;
	float* v_res = k_res + kv_size;
	float* gemm_res = v_res + kv_size;
	float* softmax_acc = gemm_res + gemm_res_size;

	const size_t m = mhsa.config.input_sequence_size;
	const size_t n = d_wq.cols;

	// One thread per element of the output
	// One thread block per 32x32 submatrix of the output
	// (32x512)*(512x512)=(32x512)
	dim3 spmm_block_gm(32, 32);
	dim3 spmm_grid_gm(
		(n + spmm_block_gm.x - 1) / spmm_block_gm.x,
		(m + spmm_block_gm.y - 1) / spmm_block_gm.y);

	// One thread per element of the output.
	// One thread block stretched across a row of the output
	// (32x512)*(512x512)=(32x512)
	dim3 spmm_block_sm(512);
	dim3 spmm_grid_sm(32);

	// One thread per element of the output.
	// One thread block stretched across a row of the output
	// (32x512)*(512x32)=(32x32)
	dim3 gemm_block_sm(32);
	dim3 gemm_grid_sm(32);

	// One thread per element of the output.
	// One thread block per 32x32 submatrix of the output
	// (32x32)
	dim3 softmax_block(32, 32);
	dim3 softmax_grid(
		(m + softmax_block.x - 1) / softmax_block.x,
		(m + softmax_block.y - 1) / softmax_block.y);  // This should actually be equal to (1,1) i.e. one block

#if defined(__CHRONO__)
	cudaEvent_t start, stop;
	float       time;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start, 0));
#endif

	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(x, d_wq.col_ptr, d_wq.row_idx, d_wq.val, mhsa.config.input_sequence_size, d_wq.rows, d_wq.cols, q_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(x, d_wk.col_ptr, d_wk.row_idx, d_wk.val, mhsa.config.input_sequence_size, d_wk.rows, d_wk.cols, k_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(x, d_wv.col_ptr, d_wv.row_idx, d_wv.val, mhsa.config.input_sequence_size, d_wv.rows, d_wv.cols, v_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	gemm<<<gemm_grid_sm, gemm_block_sm>>>(q_res, k_res, mhsa.config.input_sequence_size, d_wq.rows, mhsa.config.input_sequence_size, gemm_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	// TODO: Don't write back to gemm_res
	softmax<<<softmax_grid, softmax_block>>>(gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, softmax_acc, gemm_res);

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
	CUDA_CHECK(cudaMemcpy(mhsa.host, q_res, sizeof(float) * kv_size * 3 + gemm_res_size, cudaMemcpyDeviceToHost));

	cuda_dealloc_device(dev);
}
