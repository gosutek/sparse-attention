#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>

#include "mma.h"

#include "Utils.h"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

[[maybe_unused]] static void load_binary_to_host(const std::filesystem::path& filepath)
{
	void* host_ptr = nullptr;
	// TODO: Check if its actually a file (???)
	size_t filesize = std::filesystem::file_size(filepath);

	cudaMallocHost(&host_ptr, filesize);

	// TODO: Add error handling
	FILE* file = fopen(filepath.c_str(), "rb");
	fread(host_ptr, filesize, 1, file);
	fclose(file);

	MatrixHeader* header_ptr = reinterpret_cast<MatrixHeader*>(host_ptr);

	printf("rows: %d\n", header_ptr->rows);
	printf("cols: %d\n", header_ptr->cols);
	printf("nnz: %ld\n", header_ptr->nnz);

	// WARNING: This should only happen once every
	// matrix needed is loaded into device memory
	// since its heavy
	cudaFreeHost(host_ptr);
}

[[maybe_unused]] static void query_device()
{
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	printf("%d", device_prop.asyncEngineCount);
}

// TODO: Read binary file size
// TODO: Decide on how to pass the input, filename
// Header
// Sparse Data
// Dense Data
int main()
{
	// load_binary_to_host("~/projects/sparse-attention/data/scircuit.csr");
	// print_matrix_specs("/home/godot/projects/sparse-attention/data/fv1/fv1.mtx");
	const auto path = std::filesystem::directory_iterator("/home/godot/projects/sparse-attention/data/testing/");

	// query_device();
	try {
		convert(path, &write_hrpb);
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	// load_binary_to_host("/home/godot/projects/sparse-attention/data/fv1/fv1.csr");

	return 0;
}
