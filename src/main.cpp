#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include "allocator.h"
#include "utils.h"

constexpr const char* prunning_methods[] = { "l0_regularization/", "variational_dropout/", "magnitude_pruning/", "random_pruning/" };
constexpr const char* sparsity_arr[] = { "0.5/", "0.6/", "0.7/", "0.8/", "0.9/", "0.95/", "0.98/" };
constexpr const char* custom_sparse[] = { "1024/" };
constexpr const char* DEFAULT_TEST_DIR = "test/dlmc/";

// struct Benchmark
// {
// 	float  time[std::size(BENCH_DIMS)];
// 	double flops[std::size(BENCH_DIMS)];
// };
//
// void print_device_properties()
// {
// 	cudaDeviceProp dev_prop = {};
// 	CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));
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
// 		"Total Global Memory", static_cast<uint32_t>(dev_prop.totalGlobalMem / 1e6),
// 		"Max shared memory per block", static_cast<uint32_t>(dev_prop.sharedMemPerBlock / 1e3),
// 		"Max shared memory per SM", dev_prop.sharedMemPerMultiprocessor,
// 		"SM count", dev_prop.multiProcessorCount);
// }
//
// void print_help()
// {
// 	const std::string help_msg = std::format(
// 		"usage: cute [options]\n\n"
// 		"Options:\n"
// 		"\t-b <kernel number>      Benchmark a kernel, use -l [ --list ] for a list of kernel numbers.\n"
// 		"\t-l                      Enumerate kernels for use with -b.\n"
// 		"\t-m                      Run the entire pipeline.\n"
// 		"\t-p                      Print device properties.\n");
//
// 	std::cout << help_msg << "\n";
// }
//
// void list_kernels()
// {
// 	const std::string kernel_msg = "Placeholder";
//
// 	std::cout << kernel_msg << "\n";
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
// // void print_benchmarks(const float time, const uint32_t size_idx, const size_t nnz)
// // {
// // 	float  avg_time = time / BENCHMARKING_ROUNDS;
// // 	double flops = 2 * BENCHMARKING_DENSE_N_ROWS[size_idx] * nnz;
// //
// // 	std::cout << std::format(
// // 		"Number of rows: {}\n"
// // 		"Avg. time: {:.6f} s\n"
// // 		"Flops: {:.6f} GFLOPs/s\n",
// // 		BENCHMARKING_DENSE_N_ROWS[size_idx], avg_time, (BENCHMARKING_ROUNDS * flops * 1e-9) / time);
// // }
//
// Benchmark benchmark_spmm_csr(void (*run_kernel)(SPMM<CSR>&, const uint32_t), const std::string prunning_method = "l0_regularization/", const std::string sparsity = "0.5/")
// {
// 	// 1. Read weight
// 	// 2. Generate X with sizes (32, 64, 128, 256, 512)
// 	// 3. For each size
// 	// 3.1 Run once
// 	// 3.2 Verify result
// 	// 3.3 Run 100-1000 times each
// 	// 3.4 Calculate FLOPs
//
// 	SPMM<CSR>   spmm;
// 	Benchmark   res;
// 	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
// 	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
// 		spmm.sparse_path = data_dir_path + "q_fully_connected.smtx";
// 	} else {
// 		spmm.sparse_path = data_dir_path + "q.smtx";
// 	}
//
// 	prepare_spmm_mem_csr(spmm);
//
// 	float       time;
// 	cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		bool correct = warmup_spmm_csr(spmm, 0, run_kernel);
// 		cudaEventRecord(start);
// 		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
// 			run_kernel(spmm, i);
// 		}
// 		cudaEventRecord(stop);
// 		cudaEventSynchronize(start);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&time, start, stop);
//
// 		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
// 		res.flops[i] = ((2 * BENCH_DIMS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
// 	}
//
// 	cudaEventDestroy(start);
// 	cudaEventDestroy(stop);
//
// 	cuda_dealloc_host(spmm.host.data);
// 	cuda_dealloc_device(spmm.dev.data);
//
// 	return res;
// }
//
// Benchmark benchmark_spmm_csc(void (*run_kernel)(SPMM<CSC>&, const uint32_t), const std::string prunning_method, const std::string sparsity)
// {
// 	// 1. Read weights
// 	// 2. Generate X with sizes (32, 64, 128, 256, 512)
// 	// 3. For each size
// 	// 3.1 Run once
// 	// 3.2 Verify result against cuspase
// 	// 3.3 Run 100-1000 times each
// 	// 3.4 Calculate FLOPs
//
// 	SPMM<CSC>   spmm;
// 	Benchmark   res;
// 	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 5);
// 	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
// 		spmm.sparse_path = data_dir_path + "v_fully_connected.smtx";
// 	} else {
// 		spmm.sparse_path = data_dir_path + "v.smtx";
// 	}
//
// 	prepare_spmm_csc(spmm);
//
// 	float       time;
// 	cudaEvent_t start, stop;
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		bool correct = warmup_spmm_csc(spmm, 0, run_kernel);
// 		cudaDeviceSynchronize();
// 		cudaEventCreate(&start);
// 		cudaEventCreate(&stop);
// 		cudaEventRecord(start, 0);
// 		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
// 			run_kernel(spmm, i);
// 		}
// 		cudaDeviceSynchronize();
// 		cudaEventRecord(stop, 0);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&time, start, stop);
//
// 		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
// 		res.flops[i] = ((2 * BENCH_DIMS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
// 		cudaEventDestroy(start);
// 		cudaEventDestroy(stop);
// 	}
// 	cudaDeviceSynchronize();
//
// 	cuda_dealloc_host(spmm.host.data);
// 	cuda_dealloc_device(spmm.dev.data);
//
// 	return res;
// }
//
// Benchmark benchmark_cusparse(const std::string prunning_method, const std::string sparsity)
// {
// 	// WARN: This function throws but doesn't gracefuly exit!1!
// 	SPMM<CSR> spmm;
// 	Benchmark res;
//
// 	CuSparse cusparse;
// cusparseCreate(&cusparse.handle);
//
// 	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
// 	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
// 		spmm.sparse_path = data_dir_path + "v_fully_connected.smtx";
// 	} else {
// 		spmm.sparse_path = data_dir_path + "v.smtx";
// 	}
//
// 	prepare_cusparse_csr(spmm, cusparse);
//
// 	float       time;
// 	cudaEvent_t start, stop;
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		// Warmup
// 		CUSPARSE_CHECK(cusparseSpMM_preprocess(cusparse.handle,
// 			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
// 			&cusparse.alpha, cusparse.sparse, cusparse.dense[0], &cusparse.beta, cusparse.res[0],
// 			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
//
// 		CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
// 			CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
// 			&cusparse.alpha, cusparse.sparse, cusparse.dense[0], &cusparse.beta, cusparse.res[0], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
//
// 		cudaDeviceSynchronize();
// 		cudaEventCreate(&start);
// 		cudaEventCreate(&stop);
// 		cudaEventRecord(start, 0);
// 		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
// 			CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
// 				CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
// 				&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
// 		}
// 		cudaDeviceSynchronize();
// 		cudaEventRecord(stop, 0);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&time, start, stop);
// 		cudaEventDestroy(start);
// 		cudaEventDestroy(stop);
//
// 		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
// 		res.flops[i] = ((2 * BENCH_DIMS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
// 	}
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	cuda_dealloc_device(cusparse.work_buffer);
//
// 	cusparseDestroySpMat(cusparse.sparse);
//
// 	for (size_t i = 0; i < std::size(BENCH_DIMS); ++i) {
// 		cusparseDestroyDnMat(cusparse.dense[i]);
// 		cusparseDestroyDnMat(cusparse.res[i]);
// 	}
// 	cusparseDestroy(cusparse.handle);
//
// 	cuda_dealloc_host(spmm.host.data);
// 	cuda_dealloc_device(spmm.dev.data);
//
// 	return res;
// }

int main(int argc, char* argv[])
{
	std::vector<std::filesystem::path> input_files;
	if (argc < 2) {
		std::cout << std::format("No directory given, falling back to default: '{}'\n", DEFAULT_TEST_DIR);

		const std::filesystem::path arg_dir(DEFAULT_TEST_DIR);
		input_files = collect_rec_input(arg_dir);

	} else if (argc == 2) {  // For every '.smtx' in the dir, spmm with a generated dense
		const std::filesystem::path arg_dir(argv[1]);
		if (!std::filesystem::exists(arg_dir)) {
			std::cout << std::format("File/Directory given does not exist: '{}'.\nExiting...\n", arg_dir.string());
			return -1;
		}
		if (std::filesystem::is_directory(arg_dir)) {
			// TODO: make a better print
			std::cout << std::format("Directory given.\n");
			input_files = collect_rec_input(arg_dir);
		} else if (std::filesystem::is_regular_file(arg_dir)) {
			if (arg_dir.extension() != ".smtx") {
				std::cout << std::format("Non '.smtx' file given: '{}'\n", arg_dir.string());
			}
			std::cout << std::format("File given.\n");
		} else {
			std::cout << std::format("Unknown path: '{}'\n", arg_dir.string());
			return -1;
		}
	} else if (argc == 3) {  //
		std::cout << std::format("Not implemented yet\n");
		return -1;
	} else {
		std::cout << std::format("Give either the directory or path (sparse LOR dense)\n");
		return -1;
	}

	// for (int i = 1; i < argc; ++i) {
	// 	if (argv[i][0] != '-') {
	// 		print_help();
	// 		return EXIT_FAILURE;
	// 	}
	// 	if (strlen(argv[i]) != 2) {
	// 		print_help();
	// 		return EXIT_FAILURE;
	// 	}
	// 	if (argv[i][1] == 'b') {
	// 		if (i + 1 >= argc) {
	// 			print_help();
	// 			return EXIT_FAILURE;
	// 		}
	//
	// 		int kernel = std::atoi(argv[i + 1]);
	// 		++i;
	//
	// 		SPMM<CSR> spmm;
	//
	// 		std::string data_dir_path = construct_path("data/dlmc/transformer/l0_regularization/0.5", BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
	// 		spmm.sparse_path = data_dir_path + "v.smtx";
	//
	// 		Benchmark sota;
	// 		Benchmark custom;
	//
	// 		/**
	//      *    +------------------------------------------------------------------------------------------+
	//      *    |                                       ALLOCATING                                         |
	//      *    +------------------------------------------------------------------------------------------+
	//      */
	// 		prepare_spmm_mem_csr(spmm);
	//
	// 		switch (kernel) {
	// 		case 1:
	// 			{
	// 				CuSparse cusparse;
	// 				cusparseCreate(&cusparse.handle);
	// 				prepare_cusparse_csr(spmm, cusparse);
	// 				break;
	// 			}
	// 		default:
	// 			{
	// 				print_help();
	// 				return EXIT_FAILURE;
	// 			}
	// 		}
	// 		/**
	//      *    +------------------------------------------------------------------------------------------+
	//      *    |                                     DEALLOCATING                                         |
	//      *    +------------------------------------------------------------------------------------------+
	//      */
	// 		if (spmm.host.data) {
	// 			cuda_dealloc_host(spmm.host.data);
	// 		}
	// 		if (spmm.dev.data) {
	// 			cuda_dealloc_device(spmm.dev.data);
	// 		}
	// 	} else if (argv[i][1] == 'l') {
	// 		list_kernels();
	// 	} else if (argv[i][1] == 'm') {
	// 		// Run the entire pipeline
	// 		// MHSA<CSC, CSR> mhsa;
	// 		//
	// 		// run_mhsa(mhsa);
	// 		// cuda_dealloc_host(mhsa.host.data);
	// 		// cuda_dealloc_device(mhsa.dev.data);
	// 	} else if (argv[i][1] == 'p') {
	// 		print_device_properties();
	// 	} else if (argv[i][1] == 't') {
	// 		if (i + 1 >= argc) {
	// 			print_help();
	// 			return EXIT_FAILURE;
	// 		}
	//
	// 		int kernel = std::atoi(argv[i + 1]);
	// 		++i;
	//
	// 		switch (kernel) {
	// 		case 1:
	// 			std::vector<float> sparse_rm = read_row_major_from_rm({ "test/3by3" }, 9);  // pointer copy
	// 			break;
	// 		}
	// 	}
	// }

	return 0;
}
