#include <cusparse.h>
#include <format>
#include <iostream>
#include <vector>

#include "handle.h"
#include "matrix.h"
#include "spmm.cuh"

constexpr const char* prunning_methods[] = { "l0_regularization/", "variational_dropout/", "magnitude_pruning/", "random_pruning/" };
constexpr const char* sparsity_arr[] = { "0.5/", "0.6/", "0.7/", "0.8/", "0.9/", "0.95/", "0.98/" };

struct Benchmark
{
	float  time[std::size(BENCHMARKING_DENSE_N_ROWS)];
	double flops[std::size(BENCHMARKING_DENSE_N_ROWS)];
};

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
		"List of kernels for benchmarking:\n\n"
		"1. cuSparse\n"
		"2. 'spmm_naive_elemwise_csc_gmem'\n"
		"3. 'spmm_naive_elemwise_csc_smem'\n"
		"4. 'spmm_coalesced_elemwise_csr'\n"
		"5. 'spmm_blocktiling_elemwise_csr'\n"
		"6. 'spmm_coalesced_nnzwise'\n"
		"7. 'spmm_vectorized_nnzwise_smem'\n"
		"8. 'spmm_vectorized_nnzwise_regs'\n"
		"9. 'softmax'\n";

	std::cout << kernel_msg << "\n";
}

void print_benchmarks(const std::string op_name, const std::string contender_1, const std::string contender_2,
	const std::string prunning_method, const std::string sparsity,
	const Benchmark cusparse, const Benchmark custom)
{
	// mxkxn
	std::vector<std::string> rows;
	rows.reserve(std::size(BENCHMARKING_DENSE_N_ROWS));
	const size_t padding = 20;
	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		std::string shape = std::format("({:<3}, {}, {})", BENCHMARKING_DENSE_N_ROWS[i], MAT_SIZE, MAT_SIZE);
		std::string cu_time_str = std::format("{:.4f}", cusparse.time[i] * 1e3);
		std::string custom_time_str = std::format("{:.4f}", custom.time[i] * 1e3);
		std::string relative_str = std::format("{:.0f}%", (cusparse.time[i] / custom.time[i] * 100));
		std::string cu_flops_str = std::format("{:.4f}", cusparse.flops[i]);
		std::string custom_flops_str = std::format("{:.4f}", custom.flops[i]);
		std::string row = std::format("{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}\n", shape, padding, cu_time_str, padding, custom_time_str, padding, relative_str, padding, cu_flops_str, padding, custom_flops_str, padding);

		rows.push_back(row);
	}
	std::string title = std::format("{}: {} ~ {}\n", op_name, contender_1, contender_2);
	std::string input = std::format("{} | {}\n", prunning_method, sparsity);
	std::string header = std::format("{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}{:<{}}\n",
		"Shape MxKxN", padding,
		contender_1 + " (ms)", padding,
		"Custom (ms)", padding,
		"Perf ratio", padding,
		contender_1 + " GFLOPs/s", padding,
		"Custom GFLOPs/s", padding);
	std::string separator(header.size(), '-');

	std::cout << title << input << "\n"
			  << header
			  << separator << "\n";

	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		std::cout << rows[i];
	}

	std::cout << separator << "\n";
}

// void print_benchmarks(const float time, const uint32_t size_idx, const size_t nnz)
// {
// 	float  avg_time = time / BENCHMARKING_ROUNDS;
// 	double flops = 2 * BENCHMARKING_DENSE_N_ROWS[size_idx] * nnz;
//
// 	std::cout << std::format(
// 		"Number of rows: {}\n"
// 		"Avg. time: {:.6f} s\n"
// 		"Flops: {:.6f} GFLOPs/s\n",
// 		BENCHMARKING_DENSE_N_ROWS[size_idx], avg_time, (BENCHMARKING_ROUNDS * flops * 1e-9) / time);
// }

Benchmark benchmark_spmm_csr(void (*run_kernel)(SPMM<CSR>&, const uint32_t), const std::string prunning_method = "l0_regularization/", const std::string sparsity = "0.5/")
{
	// 1. Read weight
	// 2. Generate X with sizes (32, 64, 128, 256, 512)
	// 3. For each size
	// 3.1 Run once
	// 3.2 Verify result
	// 3.3 Run 100-1000 times each
	// 3.4 Calculate FLOPs

	SPMM<CSR>   spmm;
	Benchmark   res;
	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
		spmm.sparse_path = data_dir_path + "q_fully_connected.smtx";
	} else {
		spmm.sparse_path = data_dir_path + "q.smtx";
	}

	prepare_spmm_csr(spmm);

	float       time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		bool correct = warmup_spmm_csr(spmm, 0, run_kernel);
		cudaEventRecord(start);
		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
			run_kernel(spmm, i);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
		res.flops[i] = ((2 * BENCHMARKING_DENSE_N_ROWS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cuda_dealloc_host(spmm.host.data);
	cuda_dealloc_device(spmm.dev.data);

	return res;
}

Benchmark benchmark_spmm_csc(void (*run_kernel)(SPMM<CSC>&, const uint32_t), const std::string prunning_method, const std::string sparsity)
{
	// 1. Read weights
	// 2. Generate X with sizes (32, 64, 128, 256, 512)
	// 3. For each size
	// 3.1 Run once
	// 3.2 Verify result against cuspase
	// 3.3 Run 100-1000 times each
	// 3.4 Calculate FLOPs

	SPMM<CSC>   spmm;
	Benchmark   res;
	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
		spmm.sparse_path = data_dir_path + "q_fully_connected.smtx";
	} else {
		spmm.sparse_path = data_dir_path + "q.smtx";
	}

	prepare_spmm_csc(spmm);

	float       time;
	cudaEvent_t start, stop;

	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		bool correct = warmup_spmm_csc(spmm, 0, run_kernel);
		cudaDeviceSynchronize();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
			run_kernel(spmm, i);
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);

		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
		res.flops[i] = ((2 * BENCHMARKING_DENSE_N_ROWS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	cudaDeviceSynchronize();

	cuda_dealloc_host(spmm.host.data);
	cuda_dealloc_device(spmm.dev.data);

	return res;
}

Benchmark benchmark_cusparse(const std::string prunning_method, const std::string sparsity)
{
	// WARN: This function throws but doesn't gracefuly exit!1!
	SPMM<CSC> spmm;
	Benchmark res;

	CuSparse cusparse;
	cusparseCreate(&cusparse.handle);

	std::string data_dir_path = construct_path("data/dlmc/transformer/" + prunning_method + sparsity, BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
	if (prunning_method == "random_pruning/" || prunning_method == "magnitude_pruning/") {
		spmm.sparse_path = data_dir_path + "q_fully_connected.smtx";
	} else {
		spmm.sparse_path = data_dir_path + "q.smtx";
	}

	// WARN: Calling both of these is necessary at the moment but does double work.
	prepare_spmm_csc(spmm);
	prepare_cusparse_csc(spmm, cusparse);

	float       time;
	cudaEvent_t start, stop;

	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		// Warmup
		CUSPARSE_CHECK(cusparseSpMM_preprocess(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[0], &cusparse.beta, cusparse.res[0],
			CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));

		CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
			CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
			&cusparse.alpha, cusparse.sparse, cusparse.dense[0], &cusparse.beta, cusparse.res[0], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));

		cudaDeviceSynchronize();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
			CUSPARSE_CHECK(cusparseSpMM(cusparse.handle,
				CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
				&cusparse.alpha, cusparse.sparse, cusparse.dense[i], &cusparse.beta, cusparse.res[i], CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, cusparse.work_buffer));
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		res.time[i] = (time * 1e-3) / BENCHMARKING_ROUNDS;
		res.flops[i] = ((2 * BENCHMARKING_DENSE_N_ROWS[i] * spmm.host.s.nnz) * BENCHMARKING_ROUNDS * 1e-9) / (time * 1e-3);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	cuda_dealloc_device(cusparse.work_buffer);

	cusparseDestroySpMat(cusparse.sparse);

	for (size_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		cusparseDestroyDnMat(cusparse.dense[i]);
		cusparseDestroyDnMat(cusparse.res[i]);
	}
	cusparseDestroy(cusparse.handle);

	cuda_dealloc_host(spmm.host.data);
	cuda_dealloc_device(spmm.dev.data);

	return res;
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		print_help();
		return EXIT_FAILURE;
	}

	for (int i = 1; i < argc; ++i) {
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

			Benchmark sota;
			Benchmark custom;

			switch (kernel) {
			case 1:
				for (const auto& prunning_method : prunning_methods) {
					for (const auto& sparsity : sparsity_arr) {
						sota = benchmark_cusparse(prunning_method, sparsity);
					}
				}
				break;
			case 2:
				for (const auto& prunning_method : prunning_methods) {
					for (const auto& sparsity : sparsity_arr) {
						sota = benchmark_cusparse(prunning_method, sparsity);
						custom = benchmark_spmm_csc(&run_spmm_naive_elemwise_csc_gmem, prunning_method, sparsity);

						print_benchmarks("Spmm", "SOTA", "Naive elementwise CSC GMEM", prunning_method, sparsity, sota, custom);
					}
				}

				break;
			case 3:
				sota = benchmark_cusparse();
				custom = benchmark_spmm_csc(&run_spmm_naive_elemwise_csc_smem);

				print_benchmarks("Spmm", "SOTA", "Naive elementwise CSC SMEM", "l0_regularization/", "0.5/", sota, custom);
				break;
			case 4:
				sota = benchmark_cusparse();
				custom = benchmark_spmm_csr(&run_spmm_coalesced_elemwise_csr);

				print_benchmarks("Spmm", "SOTA", "Coalesced elementwise CSR", "l0_regularization/", "0.5/", sota, custom);
				break;
			case 5:
				sota = benchmark_cusparse();
				custom = benchmark_spmm_csr(&run_spmm_blocktiling_elemwise_csr);

				print_benchmarks("Spmm", "SOTA", "Blocktiling elementwise CSR", "l0_regularization/", "0.5/", sota, custom);
				break;
				break;
			case 6:
				sota = benchmark_cusparse();
				custom = benchmark_spmm_csc(&run_spmm_coalesced_nnzwise);

				print_benchmarks("Spmm", "SOTA", "Coalesced nonzero-wise", "l0_regularization/", "0.5/", sota, custom);
				break;
			case 7:
				for (const auto& prunning_method : prunning_methods) {
					for (const auto& sparsity : sparsity_arr) {
						sota = benchmark_cusparse(prunning_method, sparsity);
						custom = benchmark_spmm_csc(&run_spmm_vectorized_nnzwise_regs, prunning_method, sparsity);

						print_benchmarks("Spmm", "SOTA", "Vectorized nonzero-wise registers", prunning_method, sparsity, sota, custom);
					}
				}
				break;
			default:
				print_help();
				return EXIT_FAILURE;
			}
		} else if (argv[i][1] == 'l') {
			list_kernels();
		} else if (argv[i][1] == 'm') {
			// Run the entire pipeline
			// MHSA<CSC, CSR> mhsa;
			//
			// run_mhsa(mhsa);
			// cuda_dealloc_host(mhsa.host.data);
			// cuda_dealloc_device(mhsa.dev.data);
		} else if (argv[i][1] == 'p') {
			print_device_properties();
		}
	}

	return 0;
}
