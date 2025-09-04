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

void print_device_properties()
{
	cudaDeviceProp dev_prop = {};
	CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));

	std::cout << std::format(
		"- {:30}: {}\n"
		"- {:30}: {}.{}\n"
		"- {:30}: {}\n"
		"- {:30}: {}\n"
		"- {:30}: {}\n"
		"- {:30}: {}\n"
		"- {:30}: {}\n"
		"- {:30}: {} MB\n"
		"- {:30}: {} KB\n"
		"- {:30}: {} B\n"
		"- {:30}: {}\n",
		"Name", dev_prop.name,
		"Compute Capability", dev_prop.major, dev_prop.minor,
		"Max threads per block", dev_prop.maxThreadsPerBlock,
		"Max threads per SM", dev_prop.maxThreadsPerMultiProcessor,
		"Threads per warp", dev_prop.warpSize,
		"Max regs per block", dev_prop.regsPerBlock,
		"Max regs per SM", dev_prop.regsPerMultiprocessor,
		"Total Global Memory", static_cast<uint32_t>(dev_prop.totalGlobalMem / 1e6),
		"Max shared memory per block", static_cast<uint32_t>(dev_prop.sharedMemPerBlock / 1e3),
		"Max shared memory per SM", dev_prop.sharedMemPerMultiprocessor,
		"SM count", dev_prop.multiProcessorCount);
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

	// TODO: std::expf()
	float e = std::exp(get_elem_rm(a, k, y, x));
	atomicAdd(acc, e);

	__syncthreads();

	float val = e / *acc;
	set_elem_rm(res, k, y, x, val);
}

struct DevSpmm
{
	float *d{}, *res{};

	CSC s;
};

static DevSpmm prepare_spmm(CSC& s, float* d, size_t m, void* host)
{
	DevSpmm res;

	size_t input_b_size = s.b_size + m * sizeof(float);
	size_t res_b_size = (m * s.cols) * sizeof(float);

	dev = cuda_malloc_device(input_b_size + res_b_size);

	return res;
}

void run_spmm(std::filesystem::path& path)
{}

struct DevMHSA
{
	float *x{}, *q_res{}, *k_res{}, *v_res{}, *gemm_res{}, *softmax_acc{}, *attention{};
	// TODO: Maybe add byte sizes here...

	CSC w_q, w_k, w_v, w_o;
};

static DevMHSA prepare_mhsa(MHSA<CSC, CSR>& mhsa)
{
	DevMHSA res;

	// TODO: Find a better name
	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
	size_t gemm_res_size = mhsa.config.input_sequence_size * mhsa.config.input_sequence_size;

	size_t res_b_size = sizeof(float) * (kv_size * 4 + gemm_res_size * 2 + 1);  // Q, K, V, gemm result, float acc for softmax, Attention matrix, Final Result

	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));

	/*
      * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
      * | x | w_q | w_k | w_v | w_o | mask | Q | K | V | QK^T | ACC | A | Final Result |
      * +---+-----+-----+-----+-----+------+---+---+---+------+-----+---+--------------+
      * +-------------HOST-----------------+----------------DEVICE---------------------+
   */

	res.x = reinterpret_cast<float*>(mhsa.dev);
	size_t b_x_size = sizeof(float) * kv_size;

	char* ptr = reinterpret_cast<char*>(res.x) + b_x_size;

	// TODO: This call copy assignment operator of CSC
	// check if the custom one does what you want
	res.w_q = mhsa.weights.w_q[0];
	res.w_q.partition(ptr);
	ptr += res.w_q.b_size;

	res.w_k = mhsa.weights.w_k[0];
	res.w_k.partition(ptr);
	ptr += res.w_k.b_size;

	res.w_v = mhsa.weights.w_v[0];
	res.w_v.partition(ptr);
	ptr += res.w_v.b_size;

	res.w_o = mhsa.weights.w_o[0];
	res.w_o.partition(ptr);
	ptr += res.w_o.b_size;

	res.q_res = reinterpret_cast<float*>(ptr);
	res.k_res = res.q_res + kv_size;
	res.v_res = res.k_res + kv_size;
	res.gemm_res = res.v_res + kv_size;
	res.softmax_acc = res.gemm_res + gemm_res_size;
	res.attention = res.softmax_acc + 1;

	return res;
}

void run_mhsa(MHSA<CSC, CSR>& mhsa)
{
	DevMHSA      d = prepare_mhsa(mhsa);
	const size_t m = mhsa.config.input_sequence_size;
	const size_t n = d.w_q.cols;

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

	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_q.col_ptr, d.w_q.row_idx, d.w_q.val, mhsa.config.input_sequence_size, d.w_q.rows, d.w_q.cols, d.q_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::RM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_k.col_ptr, d.w_k.row_idx, d.w_k.val, mhsa.config.input_sequence_size, d.w_k.rows, d.w_k.cols, d.k_res);
	spmm_csc<KernelType::SharedMemory, OutputFormat::CM><<<spmm_grid_sm, spmm_block_sm>>>(d.x, d.w_v.col_ptr, d.w_v.row_idx, d.w_v.val, mhsa.config.input_sequence_size, d.w_v.rows, d.w_v.cols, d.v_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	gemm<<<gemm_grid_sm, gemm_block_sm>>>(d.q_res, d.k_res, mhsa.config.input_sequence_size, d.w_q.rows, mhsa.config.input_sequence_size, d.gemm_res);

	CUDA_CHECK(cudaDeviceSynchronize());

	softmax<<<softmax_grid, softmax_block>>>(d.gemm_res, mhsa.config.input_sequence_size, mhsa.config.input_sequence_size, d.softmax_acc, d.attention);

	CUDA_CHECK(cudaDeviceSynchronize());

	// TODO: can this be async?
	// TODO: THIS NEEDS TO WRITE TO PAGE-LOCKED MEMORY NOT SOME RANDOM ALLOCATED MEMORY
	//
	// CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
}
