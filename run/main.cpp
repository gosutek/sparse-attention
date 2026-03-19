#include <filesystem>
#include <iostream>

#include "cusparse.h"
#include "helpers.h"
#include "launcher.h"
#include "utils.h"

#include "spmm.h"

// struct Benchmark
// {
// 	f32  time[std::size(BENCH_DIMS)];
// 	f64 flops[std::size(BENCH_DIMS)];
// };
//
// void print_device_properties()
// {
// 	cudaDeviceProp dev_prop = {};
// 	CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, 0));
//
// 	std::cout << std::format(
// 		"- {:30}: {}\n"
// 		"- {:30}: {}.{}\n"
// 		"- {:30}: {}\n"
// 		"- {:30}: {}\n"
// 		"- {:30}: {}\n"
// 		"- {:30}: {}\n"
// 		"- {:30}: {}\n"
// 		"- {:30}: {} MB\n"
// 		"- {:30}: {} KB\n"
// 		"- {:30}: {} B\n"
// 		"- {:30}: {}\n",
// 		"Name", dev_prop.name,
// 		"Compute Capability", dev_prop.major, dev_prop.minor,
// 		"Max threads per block", dev_prop.maxThreadsPerBlock,
// 		"Max threads per SM", dev_prop.maxThreadsPerMultiProcessor,
// 		"Threads per warp", dev_prop.warpSize,
// 		"Max regs per block", dev_prop.regsPerBlock,
// 		"Max regs per SM", dev_prop.regsPerMultiprocessor,
// 		"Total Global Memory", static_cast<u32>(dev_prop.totalGlobalMem / 1e6),
// 		"Max shared memory per block", static_cast<u32>(dev_prop.sharedMemPerBlock / 1e3),
// 		"Max shared memory per SM", dev_prop.sharedMemPerMultiprocessor,
// 		"SM count", dev_prop.multiProcessorCount);
// }
//
// void print_benchmarks(const std::string op_name, const std::string contender_1, const std::string contender_2,
// 	const std::string prunning_method, const std::string sparsity,
// 	const Benchmark cusparse, const Benchmark custom)
// {
// 	// mxkxn
// 	std::vector<std::string> rows;
// 	rows.reserve(std::size(BENCH_DIMS));
// 	const size_t padding = 20;
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		std::string shape = std::format("({:<3}, {}, {})", BENCH_DIMS[i], MAT_SIZE, MAT_SIZE);
// 		std::string cu_time_str = std::format("{:.4f}", cusparse.time[i] * 1e3);
// 		std::string custom_time_str = std::format("{:.4f}", custom.time[i] * 1e3);
// 		std::string relative_str = std::format("{:.0f}%", (cusparse.time[i] / custom.time[i] * 100));
// 		std::string cu_flops_str = std::format("{:.4f}", cusparse.flops[i]);
// 		std::string custom_flops_str = std::format("{:.4f}", custom.flops[i]);
// 		std::string row = std::format("{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}\n", shape, padding, cu_time_str, padding, custom_time_str, padding, relative_str, padding, cu_flops_str, padding, custom_flops_str, padding);
//
// 		rows.push_back(row);
// 	}
// 	std::string title = std::format("{}: {} ~ {}\n", op_name, contender_1, contender_2);
// 	std::string input = std::format("{} | {}\n", prunning_method, sparsity);
// 	std::string header = std::format("{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}\n",
// 		"Shape MxKxN", padding,
// 		contender_1 + " (ms)", padding,
// 		"Custom (ms)", padding,
// 		"Perf ratio", padding,
// 		contender_1 + " GFLOPs/s", padding,
// 		"Custom GFLOPs/s", padding);
// 	std::string separator(header.size(), '-');
//
// 	std::cout << title << input << "\n"
// 			  << header
// 			  << separator << "\n";
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		std::cout << rows[i];
// 	}
//
// 	std::cout << separator << "\n";
// }
//
// // void print_benchmarks(const f32 time, const u32 size_idx, const size_t nnz)
// // {
// // 	f32  avg_time = time / BENCHMARKING_ROUNDS;
// // 	f64 flops = 2 * BENCHMARKING_DENSE_N_ROWS[size_idx] * nnz;
// //
// // 	std::cout << std::format(
// // 		"Number of rows: {}\n"
// // 		"Avg. time: {:.6f} s\n"
// // 		"Flops: {:.6f} GFLOPs/s\n",
// // 		BENCHMARKING_DENSE_N_ROWS[size_idx], avg_time, (BENCHMARKING_ROUNDS * flops * 1e-9) / time);
// // }

static std::vector<f32> host_spmm_rm_cm(const std::vector<f32>& a, const std::vector<f32>& b, size_t m, size_t k, size_t n)
{
	std::vector<f32> res;
	res.reserve(m * n);

	for (size_t a_row = 0; a_row < m; ++a_row) {
		for (size_t b_col = 0; b_col < n; ++b_col) {
			f32 acc = 0;
			for (size_t i = 0; i < k; ++i) {
				acc += a[a_row * k + i] * b[b_col * k + i];
			}
			res.push_back(acc);
		}
	}

	return res;
}

static void bench_spmm_cusparse(const std::filesystem::path& sp_path, const SpmmKernelType_t kernel_t, const SpmmInvert_t invert_t, const u32 bench_rounds = 1000)
{
	ExecutionContext_t spmm_handle = NULL;
	CHECK_SPMM(exec_ctx_create(&spmm_handle));

	SpmmContext ctx_spmm = setup_spmm(spmm_handle, sp_path);

	cusparseHandle_t cusparse_handle = NULL;
	CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

	CusparseContext ctx_cusparse = setup_cusparse(cusparse_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res);

	CHECK_SPMM(spmm(spmm_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res, kernel_t, invert_t));
	u32  rows_res, cols_res;
	f32* val_res;
	CHECK_SPMM(dn_rm_get(ctx_spmm.d_res, &rows_res, &cols_res, &val_res));
	CHECK_CUDA(cudaMemcpy(ctx_spmm.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&ctx_cusparse.alpha, ctx_cusparse.d_csr, ctx_cusparse.d_dn, &ctx_cusparse.beta, ctx_cusparse.d_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_cusparse.buffer));

	CHECK_CUDA(cudaMemcpy(ctx_cusparse.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	for (u32 i = 0; i < rows_res * cols_res; ++i) {
		comparef(ctx_spmm.h_res[i], ctx_cusparse.h_res[i]);
	}

	f32         time;
	cudaEvent_t start, stop;
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (u32 i = 0; i < bench_rounds; ++i) {
		CHECK_SPMM(spmm(spmm_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res, kernel_t, invert_t));
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	const f64 custom_time = time / bench_rounds;  // ms
	const f64 custom_flops = (2.0 * ctx_spmm.h_csr.nnz * cols_res * 1e-9) / (custom_time * 1e-3);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (u32 i = 0; i < bench_rounds; ++i) {
		CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ctx_cusparse.alpha, ctx_cusparse.d_csr, ctx_cusparse.d_dn, &ctx_cusparse.beta, ctx_cusparse.d_res, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_cusparse.buffer));
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	const f64 cusparse_time = time / bench_rounds;
	const f64 cusparse_flops = (2.0 * ctx_spmm.h_csr.nnz * cols_res * 1e-9) / (cusparse_time * 1e-3);

	std::cout << "Avg. time: " << custom_time << " | " << cusparse_time << " ms\nFlops: " << custom_flops << " | " << cusparse_flops << " GFLOPs/s\n";
	// std::cout << std::format(
	// 	"Avg. time: {:.6f} | {:.6f} s\n"
	// 	"Flops: {:.6f} | {:.6f} GFLOPs/s\n",
	// 	custom_time, cusparse_time, custom_flops, cusparse_flops);

	CHECK_SPMM(exec_ctx_destroy(spmm_handle));

	CHECK_CUSPARSE(cusparseDestroySpMat(ctx_cusparse.d_csr));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_cusparse.d_dn));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_cusparse.d_res));
	CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

int main(void)
{
	// const std::filesystem::directory_iterator dir_it("run/data/dlmc/transformer/l0_regularization/0.5/");
	// for (const std::filesystem::path& p : dir_it) {
	// 	if (!p.stem().string().ends_with("aux")) {
	// 		std::cout << "Benchmarking: " << p.stem().string() << std::endl;
	// 		bench_spmm_cusparse(p, SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK, SPMM_KERNEL_NO_INVERT);
	// 	}
	// }
	bench_spmm_cusparse("run/data/dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_self_attention_multihead_attention_q.smtx", SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK, SPMM_KERNEL_NO_INVERT);
	return 0;
}
