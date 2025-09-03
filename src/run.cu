#include <cusparse.h>

#include "common.h"
#include "matrix.h"
#include "model.h"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUSPARSE_CHECK(x)                                                                                    \
	do {                                                                                                     \
		cusparseStatus_t err = x;                                                                            \
		if (err != CUSPARSE_STATUS_SUCCESS) {                                                                \
			fprintf(stderr, "CUSPARSE error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cusparseGetErrorString(err), cusparseGetErrorName(err), err);                                \
			abort();                                                                                         \
		}                                                                                                    \
	} while (0)

void  print_device_properties();
void  cuda_dealloc_host(void* ptr);
void  cuda_dealloc_device(void* ptr);
void* cuda_malloc_device(size_t b_size);
void  run(MHSA<CSC, CSR>& mhsa, float* res);
void  test_dev_spmm();
bool  verify_res(const float* const actual, const float* const expected, size_t n);

void run_cusparse_spmm(cusparseHandle_t handle, void* col_ptr, void* row_idx, void* val,
	size_t m, size_t k, size_t n, size_t nnz, void* x, void* res, float alpha, float beta)
{
	cusparseSpMatDescr_t a;
	CUSPARSE_CHECK(cusparseCreateCsc(&a,
		k, n, nnz,
		col_ptr, row_idx, val,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	cusparseDnMatDescr_t b, c;
	// CUSPARSE_CHECK(cusparseCreateDnMat(&b, k, m, k, x, CUDA_R_32F, CUSPARSE_ORDER_COL));
	// CUSPARSE_CHECK(cusparseCreateDnMat(&c, n, m, n, res, CUDA_R_32F, CUSPARSE_ORDER_COL));

	CUSPARSE_CHECK(cusparseCreateDnMat(&b, m, k, k, x, CUDA_R_32F, CUSPARSE_ORDER_ROW));
	CUSPARSE_CHECK(cusparseCreateDnMat(&c, n, m, n, res, CUDA_R_32F, CUSPARSE_ORDER_COL));

	size_t work_buffer_size = 0;
	CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&alpha, a, b, &beta, c,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &work_buffer_size));

	void* work_buffer = cuda_malloc_device(work_buffer_size);

	CUSPARSE_CHECK(cusparseSpMM_preprocess(handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&alpha, a, b, &beta, c,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, work_buffer));

#if defined(__CHRONO__)
	cudaEvent_t start, stop;
	float       time;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start, 0));
#endif

	CUSPARSE_CHECK(cusparseSpMM(handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&alpha, a, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, work_buffer));

#if defined(__CHRONO__)
	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));

	CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	std::cout << std::format("cuSparse kernel: {} ms\n", time);
#endif

	cuda_dealloc_device(work_buffer);
}

void run_spmm(MHSA<CSC, CSR>& mhsa, float* res)
{
	size_t kv_size = mhsa.config.input_sequence_size * MAT_SIZE;  // k OR v's size
	size_t res_b_size = sizeof(float) * kv_size;
	mhsa.dev = cuda_malloc_device(mhsa.b_size + res_b_size);
	CUDA_CHECK(cudaMemcpy(mhsa.dev, mhsa.host, mhsa.b_size, cudaMemcpyHostToDevice));

	float* x = reinterpret_cast<float*>(mhsa.dev);
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

	const size_t m = mhsa.config.input_sequence_size;
	const size_t k = d_wq.rows;
	const size_t n = d_wq.cols;

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	run_cusparse_spmm(handle, d_wq.col_ptr, d_wq.row_idx, d_wq.val, m, k, n, d_wq.nnz, x, q_res, 1, 0);
	cusparseDestroy(handle);

	CUDA_CHECK(cudaDeviceSynchronize());

	// TODO: can this be async?
	CUDA_CHECK(cudaMemcpy(res, q_res, sizeof(float) * kv_size, cudaMemcpyDeviceToHost));
}

void print_x(float* x, size_t size)
{
	for (size_t i = 0; i < 5; ++i) {
		std::cout << std::format("x[{}]: {}\n", i, x[i]);
	}
	std::cout << "---------------------------------------------------" << std::endl;
}

void print_help()
{
	const std::string help_msg = std::format(
		"usage: cute [options]\n\n"
		"Options:\n"
		"\t-b <kernel number>      Benchmark a kernel, use -l [ --list ] for a list of kernel numbers.\n"
		"\t-l                      Enumerate kernels for use with -b.\n"
		"\t-m                      Run the entire pipeline.\n"
		"\t-p                      Print device properties.\n");

	std::cout << help_msg << "\n";
}

void list_kernels()
{
	const std::string kernel_msg =
		"List of kernels:\n\n"
		"1. SpMM\n"
		"2. SDDMM\n"
		"3. SoftMax\n";

	std::cout << kernel_msg << "\n";
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		print_help();
		return EXIT_FAILURE;
	}

	for (size_t i = 1; i < argc; ++i) {
		if (argv[i][0] != '-') {
			print_help();
			return EXIT_FAILURE;
		}
		if (strlen(argv[i]) != 2) {
			print_help();
			return EXIT_FAILURE;
		}
		if (argv[i][1] == 'b') {
			if (i + 1 >= argc) {
				print_help();
				return EXIT_FAILURE;
			}

			int kernel = std::atoi(argv[i + 1]);
			++i;

			switch (kernel) {
			case 1:
				std::cout << "Benchmark SpMM\n";
				break;
			case 2:
				std::cout << "Benchmark SDDMM\n";
				break;
			case 3:
				std::cout << "Benchmark wtf\n";
				break;
			default:
				print_help();
				return EXIT_FAILURE;
			}
		} else if (argv[i][1] == 'l') {
			list_kernels();
		} else if (argv[i][1] == 'm') {
			// Run entire pipeline
		} else if (argv[i][1] == 'p') {
			print_device_properties();
		}
	}

	// MHSA<CSC, CSR> mhsa;
	//
	// const char* base_data_path = "data/dlmc/transformer/";
	// const char* s_pruning_method = "l0_regularization/";
	// const char* sparsity = "0.5/";
	//
	// load_host_csc(mhsa, mhsa.config, mhsa.weights, base_data_path, s_pruning_method, sparsity, AttentionMechanism::SelfAttention);
	//
	// float* cusparse_res = (float*)std::malloc(sizeof(float) * MAT_SIZE * mhsa.config.input_sequence_size);
	// run_spmm(mhsa, cusparse_res);
	//
	// float* cute_res = (float*)std::malloc(sizeof(float) * MAT_SIZE * mhsa.config.input_sequence_size);
	// run(mhsa, cute_res);
	//
	// verify_res(cute_res, cusparse_res, sizeof(float) * MAT_SIZE * mhsa.config.input_sequence_size);
	//
	// std::free(cusparse_res);
	// std::free(cute_res);
	// cuda_dealloc_host(mhsa.host);
	// cuda_dealloc_device(mhsa.dev);
	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
