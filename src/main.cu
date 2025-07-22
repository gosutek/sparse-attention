
#include "common.h"
#include "matrix_ops.cuh"

#include "mma.h"
#include <filesystem>

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
	// const auto binary_path = std::filesystem::current_path() / DATA_DIRECTORY / "d50_s2048/d50_s2048.spmm";
	const auto data_path = std::filesystem::current_path() / DATA_DIRECTORY / "dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_self_attention_multihead_attention_q.smtx";

	try {
		// SpmmInput spmm_input = deserialize(binary_path);
		// get_non_zero_col_predicate(spmm_input.d_pcm_sparse, spmm_input.rows, spmm_input.cols);
		// cudaFree(spmm_input.pitched_ptr);   // NOTE: This frees both sparse_pitched and dense_pitched | DATA LIVES HERE
		// cudaFree(spmm_input.d_pcm_sparse);  // NOTE: This frees both structs for prm_sparse and prm_dense | META DATA LIVES HERE

		CSRMatrix csr_matrix = dlmc_to_csr(data_path);

		std::cout << "Rows: " << csr_matrix.rows << "\nCols: "
				  << csr_matrix.cols << "\nNNZ: " << csr_matrix.nnz
				  << "\nFirst 2 elements of col_idx: " << csr_matrix.col_idx[0] << ", " << csr_matrix.col_idx[1]
				  << "\nFirst 2 elements of row_ptr: " << csr_matrix.row_ptr[0] << ", " << csr_matrix.row_ptr[1]
				  << "\nFirst 2 elements of val: " << csr_matrix.val[0] << ", " << csr_matrix.val[1] << "\n";
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
