#include <format>
#include <iostream>

#include "handle.h"
#include "matrix.h"
#include "spmm.cuh"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

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
		"List of kernels:\n\n"
		"1. SpMM\n"
		"2. SDDMM\n"
		"3. SoftMax\n";

	std::cout << kernel_msg << "\n";
}

void print_benchmarking_results()
{}

void benchmark_spmm()
{
	// 1. Read weight
	// 2. Generate X with sizes (32, 64, 128, 256, 512)
	// 3. For each size
	// 3.1 Run once
	// 3.2 Verify result
	// 3.3 Run 100-1000 times each
	// 3.4 Calculate FLOPs

	SPMM<CSC>   spmm;
	std::string data_dir_path = construct_path("data/dlmc/transformer/l0_regularization/0.5/", BodyType::Decoder, AttentionMechanism::SelfAttention, 0);
	spmm.sparse_path = data_dir_path + "q.smtx";

	prepare_spmm(spmm);

	for (uint8_t i = 0; i < std::size(BENCHMARKING_DENSE_N_ROWS); ++i) {
		warmup_spmm(spmm, 0);
		for (size_t j = 0; j < BENCHMARKING_ROUNDS; ++j) {
			run_spmm(spmm, i);
		}
	}

	// print_benchmarking_results();

	cuda_dealloc_host(spmm.host.data);
	cuda_dealloc_device(spmm.dev.data);
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

			switch (kernel) {
			case 1:
				std::cout << "Benchmark SpMM\n";
				benchmark_spmm();
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
	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
