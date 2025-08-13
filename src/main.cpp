#include "common.h"
#include "matrix.h"
#include "model.h"

void cuda_dealloc_host(void* ptr);
void run(MHSA mhsa);

int main(int argc, char* argv[])
{
	MHSA mhsa;

	const char* base_data_path = "data/dlmc/transformer/";
	const char* s_pruning_method = "l0_regularization/";
	const char* sparsity = "0.5/";

	load_host(mhsa, mhsa.config, mhsa.weights, base_data_path, s_pruning_method, sparsity, AttentionMechanism::SelfAttention);
	run(mhsa);

	cuda_dealloc_host(mhsa.host);

	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
