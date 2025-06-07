
#include "common.h"
#include "matrix_ops.cuh"

#include "mma.h"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

[[maybe_unused]] static void query_device()
{
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	printf("%d", device_prop.asyncEngineCount);
}

int main()
{
	const auto binary_path = std::filesystem::current_path() / DATA_DIRECTORY / "d50_s2048/d50_s2048.spmm";

	try {
		SpmmInput spmm_input = deserialize(binary_path);
		get_non_zero_col_predicate(spmm_input.d_pcm_sparse, spmm_input.rows, spmm_input.cols);
		cudaFree(spmm_input.pitched_ptr);   // NOTE: This frees both sparse_pitched and dense_pitched | DATA LIVES HERE
		cudaFree(spmm_input.d_pcm_sparse);  // NOTE: This frees both structs for prm_sparse and prm_dense | META DATA LIVES HERE
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
