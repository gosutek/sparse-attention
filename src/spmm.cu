#include "common.h"
#include "matrix.h"

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

static void* prepare(Input& input)
{
	// TODO: Streams go here
	void* dev = cuda_device_copy(input.data, input.b_size);
	return dev;
}

__global__ void spmm_kernel(
	const uint32_t* const row_ptr,
	const uint32_t* const col_idx,
	const float* const    val,
	const float* const    dense,  // expect col-major for coalesced access
	const uint32_t        nrows,
	const uint32_t        ncols,
	float* const          res)  // expect row-major for coalesced access
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < nrows) {
		float acc = 0;
		for (size_t i = row_ptr[y]; i < row_ptr[y + 1]; ++i) {
			printf("val[i] = %f mulled by dense[x * nrows + col_idx[i]] = %f\n", val[i], dense[x * nrows + col_idx[i]]);
			acc += val[i] * dense[x * nrows + col_idx[i]];
		}
		res[ncols * y + x] = acc;
	}
}

void run(Input input)
{
	CSRMatrix& q_weights = input.weights[0];
	float*     res = static_cast<float*>(cuda_malloc_device(sizeof(float) * 3 * 3));
	// TODO: Merge these two ^ v allocations
	void* dev = prepare(input);

	uint32_t* d_row_ptr = reinterpret_cast<uint32_t*>(dev);
	uint32_t* d_col_idx = d_row_ptr + q_weights.row_ptr_size;
	float*    d_val = reinterpret_cast<float*>(d_col_idx + q_weights.col_idx_size);
	float*    d_embeddings = d_val + q_weights.val_size;

	dim3 dimBlock(3, 3);
	dim3 dimGrid(1, 1, 1);

	spmm_kernel<<<dimGrid, dimBlock>>>(d_row_ptr, d_col_idx, d_val, d_embeddings, q_weights.rows, q_weights.cols, res);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(input.data, res, sizeof(float) * 3 * 3, cudaMemcpyDeviceToHost));

	cuda_dealloc_device(res);
	cuda_dealloc_device(dev);
}
