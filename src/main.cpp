#include "common.h"
#include "matrix.h"

void dealloc_host(void* ptr);

int main()
{
	const auto data_path = std::filesystem::current_path() / DATA_DIRECTORY / "dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_self_attention_multihead_attention_q.smtx";

	try {
		Input input = read_input(data_path);

		std::cout << "Rows: " << input.weights[0].rows << "\nCols: "
				  << input.weights[0].cols << "\nNNZ: " << input.weights[0].nnz
				  << "\nFirst 2 elements of col_idx: " << input.weights[0].col_idx[0] << ", " << input.weights[0].col_idx[1]
				  << "\nFirst 2 elements of row_ptr: " << input.weights[0].row_ptr[0] << ", " << input.weights[0].row_ptr[1]
				  << "\nFirst 2 elements of val: " << input.weights[0].val[0] << ", " << input.weights[0].val[1] << "\n";

		std::cout << "First 2 elements of embeddings: " << input.embeddings[0] << ", " << input.embeddings[1];

		dealloc_host(input.data);

	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
