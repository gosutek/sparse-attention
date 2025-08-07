#include "common.h"
#include "matrix.h"
#include "model.h"

int main(int argc, char* argv[])
{
	MHSA mhsa;

	std::string base_data_path = "data/dlmc/transformer/";
	std::string s_pruning_method = "l0_regularization/";
	std::string sparsity = "0.5/";
	std::string body = "body_decoder_";
	std::string attention_mechanism = "self_attention_multihead_attention_";
	int         n_layers = 0;

	read_input(mhsa, mhsa.weights, base_data_path, s_pruning_method, sparsity, body, attention_mechanism, n_layers);

	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
