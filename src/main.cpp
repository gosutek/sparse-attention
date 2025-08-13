#include "common.h"
#include "matrix.h"
#include "model.h"

void cuda_dealloc_host(void* ptr);

int main(int argc, char* argv[])
{
	MHSA mhsa;

	const char* base_data_path = "/data/dlmc/transformer/";
	const char* s_pruning_method = "l0_regularization/";
	const char* sparsity = "0.5/";
	const char* body = "body_decoder_";
	const char* attention_mechanism = "self_attention_multihead_attention_";
	const int   layer = 0;

	read_input(mhsa, mhsa.config, mhsa.weights, base_data_path,
		s_pruning_method, sparsity, body, attention_mechanism, layer);

	cuda_dealloc_host(mhsa.host);

	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
