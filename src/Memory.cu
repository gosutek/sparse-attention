#include <cstdio>
#include <filesystem>
#include <vector>

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
	void*  host_ptr = nullptr;
	size_t filesize = std::filesystem::file_size(filepath);

	cudaMallocHost(&host_ptr, filesize);

	// TODO: Add error handling
	FILE* file = fopen(filepath.c_str(), "rb");
	fread(host_ptr, 1, filesize, file);
	fclose(file);

	// WARNING: This should only happen once every
	// matrix needed is loaded into device memory
	// since its heavy
	cudaFreeHost(host_ptr);
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
	// convert_all();
	// load_binary_to_host("~/projects/sparse-attention/data/scircuit.csr");

	print_matrix_specs("/home/godot/projects/sparse-attention/data/fv1/fv1.mtx");
	convert(std::filesystem::directory_iterator("/home/godot/projects/sparse-attention/data/fv1"));

	return 0;
}
