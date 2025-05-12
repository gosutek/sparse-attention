#include <cstdio>
#include <filesystem>
#include <vector>

#include "mma.h"

struct MatrixHeader
{
	int32_t rows;
	int32_t cols;
	int64_t nnz;

	size_t row_ptr_bytes;
	size_t col_idx_bytes;
	size_t val_bytes;
	size_t dense_bytes;
};

// TODO: Make header file for Utils.cpp
void               convert(const std::filesystem::directory_iterator& target_dir);
void               print_matrix_specs(const std::filesystem::path& filepath);
std::vector<float> generate_dense(size_t size);

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

static void load_binary_to_host(const std::filesystem::path& filepath)
{
	void* host_ptr = nullptr;
	// TODO: Check if its actually a file.
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

static void query_device()
{
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	printf("%d", device_prop.asyncEngineCount);
}

// TODO: Read binary file size
// TODO: Decide on how to pass the input, filename
// CSRMatrix Bytes
// Header
// Data
// DenseMatrix Bytes
// Header
// Data
int main()
{
	// load_binary_to_host("~/projects/sparse-attention/data/scircuit.csr");
	// print_matrix_specs("/home/godot/projects/sparse-attention/data/fv1/fv1.mtx");

	const auto path = std::filesystem::directory_iterator("/home/godot/projects/sparse-attention/data/fv1");

	query_device();
	convert(path);

	load_binary_to_host("/home/godot/projects/sparse-attention/data/fv1/fv1.csr");

	return 0;
}
