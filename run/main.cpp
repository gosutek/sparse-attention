#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

#include "cusparse.h"
#include "helpers.h"
#include "launcher.h"
#include "utils.h"

#include "spmm.h"

struct Benchmark
{
	u32 m, k, n;
	u32 nnz;
	f64 time[2];
	f64 flops[2];
};

template <typename Kernel>
f32 time_kernel(Kernel kernel)
{
	f32         time;
	cudaEvent_t start, stop;
	cudaDeviceSynchronize();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	kernel();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time;
}

static inline f64 calc_sparse_gflops(const f32 secs, const u32 nnz, const u32 n)
{
	return (2.0 * nnz * n * 1e-9) / secs;
}

void print_device_properties()
{
	cudaDeviceProp dev_prop = {};
	CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, 0));

	std::cout
		<< "- " << std::left << std::setw(30) << "Name" << ": " << dev_prop.name << "\n"
		<< "- " << std::left << std::setw(30) << "SM count" << ": " << dev_prop.multiProcessorCount << "\n"
		<< "- " << std::left << std::setw(30) << "Total Global Memory" << ": " << static_cast<u32>(dev_prop.totalGlobalMem / 1e6) << " MB\n"
		<< "- " << std::left << std::setw(30) << "L2 Cache Size" << ": " << static_cast<u32>(dev_prop.l2CacheSize / 1e6) << " MB\n"
		<< "- " << std::left << std::setw(30) << "Compute Capability" << ": " << dev_prop.major << "." << dev_prop.minor << "\n"
		<< "- " << std::left << std::setw(30) << "Shared memory per block" << ": " << static_cast<u32>(dev_prop.sharedMemPerBlock / 1e3) << " KB\n"
		<< "- " << std::left << std::setw(30) << "Shared memory per SM" << ": " << static_cast<u32>(dev_prop.sharedMemPerMultiprocessor / 1e3) << " KB\n"
		<< "- " << std::left << std::setw(30) << "Constant memory" << ": " << static_cast<u32>(dev_prop.totalConstMem / 1e3) << " KB\n"
		<< "- " << std::left << std::setw(30) << "Warp size" << ": " << dev_prop.warpSize << "\n"
		<< "- " << std::left << std::setw(30) << "Max threads per SM" << ": " << dev_prop.maxThreadsPerMultiProcessor << "\n"
		<< "- " << std::left << std::setw(30) << "Max threads per block" << ": " << dev_prop.maxThreadsPerBlock << "\n"
		<< "- " << std::left << std::setw(30) << "Max block dimensions" << ": " << dev_prop.maxThreadsDim[0] << " x " << dev_prop.maxThreadsDim[1] << " x " << dev_prop.maxThreadsDim[2] << "\n"
		<< "- " << std::left << std::setw(30) << "Max grid dimensions" << ": " << dev_prop.maxGridSize[0] << " x " << dev_prop.maxGridSize[1] << " x " << dev_prop.maxGridSize[2] << "\n"
		<< "- " << std::left << std::setw(30) << "Max regs per block" << ": " << dev_prop.regsPerBlock << "\n"
		<< "- " << std::left << std::setw(30) << "Max regs per SM" << ": " << dev_prop.regsPerMultiprocessor << "\n";
}

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

template <typename Kernel>
static void warmup(Kernel kernel, const u32 warmup_rounds = 10)
{
	for (u32 i = 0; i < warmup_rounds; ++i) {
		kernel();
	}
}

static Benchmark bench_spmm_cusparse(const std::filesystem::path& sp_path, const SpmmKernelType_t kernel_t)
{
	ExecutionContext_t spmm_handle = NULL;
	CHECK_SPMM(exec_ctx_create(&spmm_handle));

	SpmmContext ctx_spmm = setup_spmm(spmm_handle, sp_path);

	cusparseHandle_t cusparse_handle = NULL;
	CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

	CusparseContext ctx_cusparse = setup_cusparse(cusparse_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res);

	CHECK_SPMM(spmm(spmm_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res, kernel_t, SPMM_KERNEL_NO_INVERT));
	u32  rows_res, cols_res;
	f32* val_res;
	CHECK_SPMM(dn_rm_get(ctx_spmm.d_res, &rows_res, &cols_res, &val_res));
	CHECK_CUDA(cudaMemcpy(ctx_spmm.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&ctx_cusparse.alpha, ctx_cusparse.d_csr, ctx_cusparse.d_dn, &ctx_cusparse.beta, ctx_cusparse.d_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_cusparse.buffer));

	// INFO: Kind of redundant because both SPMM and CUSPARSE write to the same result buffer in the device
	i64             ld;
	cudaDataType    type;
	cusparseOrder_t order;
	CHECK_CUSPARSE(cusparseDnMatGet(ctx_cusparse.d_res, (i64*)&rows_res, (i64*)&cols_res, &ld, (void**)&val_res, &type, &order));
	CHECK_CUDA(cudaMemcpy(ctx_cusparse.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	for (u32 i = 0; i < rows_res * cols_res; ++i) {
		comparef(ctx_spmm.h_res[i], ctx_cusparse.h_res[i]);
	}

	warmup([&]() {
		CHECK_SPMM(spmm(spmm_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res, kernel_t, SPMM_KERNEL_NO_INVERT));
	});

	std::vector<f32> ms_spmm_vec;
	std::vector<f64> gflops_spmm_vec;

	constexpr const u32 min_iter = 5;
	constexpr const u32 max_iter = 20;

	u32 curr_iter = 0;  // INFO: Needed only for debugging
	for (u32 i = 0; i < max_iter; ++i) {
		static f64 cv_spmm = 10.0;
		if (i >= min_iter && cv_spmm < 0.01) {
			break;
		}
		f32 ms_spmm = time_kernel([&]() {
			CHECK_SPMM(spmm(spmm_handle, ctx_spmm.d_csr, ctx_spmm.d_dn, ctx_spmm.d_res, kernel_t, SPMM_KERNEL_NO_INVERT));
		});
		ms_spmm_vec.push_back(ms_spmm);

		static f64 mean_spmm = 0.0;
		static f64 variance_spmm = 0.0;

		const f64 gflops_spmm = calc_sparse_gflops(ms_spmm * 1e-3, ctx_spmm.h_csr.nnz, cols_res);
		gflops_spmm_vec.push_back(gflops_spmm);
		cv_spmm = calc_cv(gflops_spmm, mean_spmm, variance_spmm, i + 1);

		++curr_iter;
	}

	assert(ms_spmm_vec.size() == curr_iter);
	assert(gflops_spmm_vec.size() == curr_iter);

	const f32 mean_ms_spmm = meanf32(ms_spmm_vec);
	const f64 mean_gflops_spmm = meanf64(gflops_spmm_vec);

	warmup([&]() {
		CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ctx_cusparse.alpha, ctx_cusparse.d_csr, ctx_cusparse.d_dn, &ctx_cusparse.beta, ctx_cusparse.d_res, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_cusparse.buffer));
	});

	std::vector<f32> ms_cusparse_vec;
	std::vector<f64> gflops_cusparse_vec;
	curr_iter = 0;
	for (u32 i = 0; i < max_iter; ++i) {
		static f64 cv_cusparse = 10.0;
		if (i >= min_iter && cv_cusparse < 0.01) {
			break;
		}
		f32 ms_cusparse = time_kernel([&]() {
			CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ctx_cusparse.alpha, ctx_cusparse.d_csr, ctx_cusparse.d_dn, &ctx_cusparse.beta, ctx_cusparse.d_res, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_cusparse.buffer));
		});
		ms_cusparse_vec.push_back(ms_cusparse);

		static f64 mean_cusparse = 0.0;
		static f64 variance_cusparse = 0.0;

		const f64 gflops_cusparse = calc_sparse_gflops(ms_cusparse * 1e-3, ctx_spmm.h_csr.nnz, cols_res);
		gflops_cusparse_vec.push_back(gflops_cusparse);

		cv_cusparse = calc_cv(gflops_cusparse, mean_cusparse, variance_cusparse, i + 1);

		++curr_iter;
	}

	assert(ms_cusparse_vec.size() == curr_iter);
	assert(gflops_cusparse_vec.size() == curr_iter);

	const f32 mean_ms_cusparse = meanf32(ms_cusparse_vec);
	const f64 mean_gflops_cusparse = meanf64(gflops_cusparse_vec);

	CHECK_SPMM(exec_ctx_destroy(spmm_handle));

	cudaFree(ctx_cusparse.buffer);
	CHECK_CUSPARSE(cusparseDestroySpMat(ctx_cusparse.d_csr));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_cusparse.d_dn));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_cusparse.d_res));
	CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));

	return {
		.m = ctx_spmm.h_csr.rows,
		.k = ctx_spmm.h_csr.cols,
		.n = cols_res,
		.nnz = ctx_spmm.h_csr.nnz,
		.time = { mean_ms_spmm, mean_ms_cusparse },
		.flops = { mean_gflops_spmm, mean_gflops_cusparse }
	};
}

static Benchmark bench_ispmm_cusparse(const std::filesystem::path& sp_path, const SpmmKernelType_t kernel_t)
{
	ExecutionContext_t spmm_handle = NULL;
	CHECK_SPMM(exec_ctx_create(&spmm_handle));

	ISpmmContext ctx_ispmm = setup_ispmm(spmm_handle, sp_path);

	cusparseHandle_t cusparse_handle = NULL;
	CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

	CusparseContext ctx_icusparse = setup_icusparse(cusparse_handle, ctx_ispmm.d_csc, ctx_ispmm.d_dn, ctx_ispmm.d_res);

	CHECK_SPMM(spmm(spmm_handle, ctx_ispmm.d_csc, ctx_ispmm.d_dn, ctx_ispmm.d_res, kernel_t, SPMM_KERNEL_INVERT));
	u32  rows_res, cols_res;
	f32* val_res;
	CHECK_SPMM(dn_rm_get(ctx_ispmm.d_res, &rows_res, &cols_res, &val_res));
	CHECK_CUDA(cudaMemcpy(ctx_ispmm.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
		CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
		&ctx_icusparse.alpha, ctx_icusparse.d_csr, ctx_icusparse.d_dn, &ctx_icusparse.beta, ctx_icusparse.d_res,
		CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_icusparse.buffer));

	// INFO: Kind of redundant because both SPMM and CUSPARSE write to the same result buffer in the device
	i64             ld;
	cudaDataType    type;
	cusparseOrder_t order;
	CHECK_CUSPARSE(cusparseDnMatGet(ctx_icusparse.d_res, (i64*)&rows_res, (i64*)&cols_res, &ld, (void**)&val_res, &type, &order));
	CHECK_CUDA(cudaMemcpy(ctx_icusparse.h_res.data(), val_res, rows_res * cols_res * sizeof *val_res, cudaMemcpyDeviceToHost));

	for (u32 i = 0; i < rows_res * cols_res; ++i) {
		comparef(ctx_ispmm.h_res[i], ctx_icusparse.h_res[i]);
	}

	warmup([&]() {
		CHECK_SPMM(spmm(spmm_handle, ctx_ispmm.d_csc, ctx_ispmm.d_dn, ctx_ispmm.d_res, kernel_t, SPMM_KERNEL_INVERT));
	});

	std::vector<f32> ms_spmm_vec;
	std::vector<f64> gflops_spmm_vec;

	constexpr const u32 min_iter = 5;
	constexpr const u32 max_iter = 20;

	u32 curr_iter = 0;  // INFO: Needed only for debugging
	for (u32 i = 0; i < max_iter; ++i) {
		static f64 cv_spmm = 10.0;
		if (i >= min_iter && cv_spmm < 0.01) {
			break;
		}
		f32 ms_spmm = time_kernel([&] {
			CHECK_SPMM(spmm(spmm_handle, ctx_ispmm.d_csc, ctx_ispmm.d_dn, ctx_ispmm.d_res, kernel_t, SPMM_KERNEL_INVERT));
		});
		ms_spmm_vec.push_back(ms_spmm);

		static f64 mean_spmm = 0.0;
		static f64 variance_spmm = 0.0;

		const f64 gflops_spmm = calc_sparse_gflops(ms_spmm * 1e-3, ctx_ispmm.h_csc.nnz, rows_res);
		gflops_spmm_vec.push_back(gflops_spmm);
		cv_spmm = calc_cv(gflops_spmm, mean_spmm, variance_spmm, i + 1);

		++curr_iter;
	}

	assert(ms_spmm_vec.size() == curr_iter);
	assert(gflops_spmm_vec.size() == curr_iter);

	const f32 mean_ms_spmm = meanf32(ms_spmm_vec);
	const f64 mean_gflops_spmm = meanf64(gflops_spmm_vec);

	warmup([&]() {
		CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&ctx_icusparse.alpha, ctx_icusparse.d_csr, ctx_icusparse.d_dn, &ctx_icusparse.beta, ctx_icusparse.d_res,
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_icusparse.buffer));
	});

	std::vector<f32> ms_cusparse_vec;
	std::vector<f64> gflops_cusparse_vec;
	curr_iter = 0;
	for (u32 i = 0; i < max_iter; ++i) {
		static f64 cv_cusparse = 10.0;
		if (i >= min_iter && cv_cusparse < 0.01) {
			break;
		}
		f32 ms_cusparse = time_kernel([&] {
			CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
				CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
				&ctx_icusparse.alpha, ctx_icusparse.d_csr, ctx_icusparse.d_dn, &ctx_icusparse.beta, ctx_icusparse.d_res,
				CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, ctx_icusparse.buffer));
		});
		ms_cusparse_vec.push_back(ms_cusparse);

		static f64 mean_cusparse = 0.0;
		static f64 variance_cusparse = 0.0;
		const f64  gflops_cusparse = calc_sparse_gflops(ms_cusparse * 1e-3, ctx_ispmm.h_csc.nnz, rows_res);
		gflops_cusparse_vec.push_back(gflops_cusparse);

		cv_cusparse = calc_cv(gflops_cusparse, mean_cusparse, variance_cusparse, i + 1);

		++curr_iter;
	}

	assert(ms_cusparse_vec.size() == curr_iter);
	assert(gflops_cusparse_vec.size() == curr_iter);

	const f32 mean_ms_cusparse = meanf32(ms_cusparse_vec);
	const f64 mean_gflops_cusparse = meanf64(gflops_cusparse_vec);

	CHECK_SPMM(exec_ctx_destroy(spmm_handle));

	cudaFree(ctx_icusparse.buffer);
	CHECK_CUSPARSE(cusparseDestroySpMat(ctx_icusparse.d_csr));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_icusparse.d_dn));
	CHECK_CUSPARSE(cusparseDestroyDnMat(ctx_icusparse.d_res));
	CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));

	return {
		.m = rows_res,
		.k = ctx_ispmm.h_csc.rows,
		.n = ctx_ispmm.h_csc.cols,
		.nnz = ctx_ispmm.h_csc.nnz,
		.time = { mean_ms_spmm, mean_ms_cusparse },
		.flops = { mean_gflops_spmm, mean_gflops_cusparse }
	};
}

// Filters:
//  1. must be ".smtx"
//  2. must not start with "symbol" -> this is the vocab
//  3. must not contain "ffn" -> these kernels are for attention not the ffn component
static inline bool is_valid_smtx(const std::filesystem::path& path)
{
	return std::filesystem::is_regular_file(path) && std::filesystem::exists(path) && path.extension().string() == ".smtx" && !path.stem().string().starts_with("symbol") && path.stem().string().find("ffn") == std::string::npos;
}

static inline i32 get_terminal_width()
{
	winsize w;
	ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	return w.ws_col;
}

static inline i32 get_terminal_height()
{
	winsize w;
	ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	return w.ws_row;
}

static void draw_bar(f32 progress, i32 width)
{
	i32 bar_width = width * 0.5;
	i32 filled = static_cast<i32>(bar_width * progress);
	i32 grey = bar_width - filled;
	i32 percent = static_cast<i32>(progress * 100);

	std::cout
		<< "\033[2K"
		<< "["
		<< "\033[38;5;33m" << std::string(filled, '.') << "\033[0m"
		<< std::string(grey, ' ')
		<< "]"
		<< " " << percent << "%";

	std::cout.flush();
}

static void spmm_log(const std::string& msg, f32 progress)
{
	i32 cols = get_terminal_width();
	i32 rows = get_terminal_height();

	std::cout << "\033[" << rows - 1 << ";1H";
	std::cout << "\033[2K";
	std::cout << msg;

	std::cout << "\033[" << rows << ";1H";
	draw_bar(progress, cols);

	std::cout.flush();
}

static void redraw_bar(float progress)
{
	i32 cols = get_terminal_width();
	i32 rows = get_terminal_height();

	std::cout << "\033[s";
	std::cout << "\033[" << rows << ";1H";
	draw_bar(progress, cols);
	std::cout << "\033[u";
	std::cout.flush();
}

void pretty_print(const std::filesystem::path& dir, const char* csv_filename)
{
	std::vector<std::filesystem::path> dataset;
	for (const std::filesystem::path& p : std::filesystem::recursive_directory_iterator(dir)) {
		if (is_valid_smtx(p)) {
			dataset.emplace_back(p);
		}
	}

	const std::filesystem::path csv_out = std::filesystem::current_path() / csv_filename;
	std::ofstream               file(csv_out, std::ios_base::out);
	file << "m,k,n,nnz,sparsity,prunning_method,custom_time,cusparse_time,custom_flops,cusparse_flops\n";

	const f32 total = static_cast<f32>(dataset.size());
	for (u32 i = 0; i < total; ++i) {
		const std::filesystem::path& p = dataset[i];
		const std::string            prunning_method = p.parent_path().parent_path().stem().string();
		const std::string            sparsity = p.parent_path().stem().string() + p.parent_path().extension().string();
		f32                          progress = static_cast<f32>(i) / total;
		spmm_log("Spmm-ing: " + p.stem().string(), progress);
		Benchmark benchmark = bench_spmm_cusparse(p, SPMM_KERNEL_TYPE_NNZWISE_COLUMN_TILING);
		// Benchmark benchmark = bench_ispmm_cusparse(p, SPMM_KERNEL_TYPE_NNZWISE_COLUMN_TILING);
		file << benchmark.m << "," << benchmark.k << "," << benchmark.n << "," << benchmark.nnz << "," << sparsity << "," << prunning_method << ","
			 << benchmark.time[0] << "," << benchmark.time[1] << "," << benchmark.flops[0] << "," << benchmark.flops[1] << "\n";

		redraw_bar(progress);
	}

	i32 rows = get_terminal_height();
	std::cout << "\033[" << rows << ";1H\033[2K";
	std::cout << "Done\n";
}

int main(void)
{
	// print_prompt_bytes("The");
	// const auto bench = bench_spmm_cusparse("run/data/dlmc/transformer/l0_regularization/0.5/body_encoder_layer_2_self_attention_multihead_attention_v.smtx", SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK);
	// std::cout << std::left << std::setw(15) << "[SPMM]: " << bench.time[0] << " ms | " << bench.flops[0] << " GFLOPs\n"
	// 		  << std::left << std::setw(15) << "[CUSPARSE]: " << bench.time[1] << " ms | " << bench.flops[1] << " GFLOPS\n";

	// const auto ibench = bench_spmm_cusparse("run/data/dlmc/transformer/random_pruning/0.5/body_encoder_layer_0_ffn_conv2_fully_connected.smtx", SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM);
	// const auto ibench = bench_spmm_cusparse("run/data/dlmc/transformer/random_pruning/0.5/body_encoder_layer_0_self_attention_multihead_attention_q_fully_connected.smtx", SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM);

	// const std::filesystem::path base_dir("run/data/dlmc/transformer/");
	// pretty_print(base_dir, "a100_spmm_nnzwise_column_tiling.csv");

	// print_device_properties();
	return 0;
}
