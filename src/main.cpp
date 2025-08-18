#include "common.h"
#include "matrix.h"
#include "model.h"

void print_device_properties();
void cuda_dealloc_host(void* ptr);
void run(CSC_MHSA mhsa);

int main(int argc, char* argv[])
{
	print_device_properties();

	CSC_MHSA mhsa;

	const char* base_data_path = "data/dlmc/transformer/";
	const char* s_pruning_method = "l0_regularization/";
	const char* sparsity = "0.5/";

	load_host_csc(mhsa, mhsa.config, mhsa.weights, base_data_path, s_pruning_method, sparsity, AttentionMechanism::SelfAttention);
	run(mhsa);

	float sparsity_perc = measure_sparsity(mhsa.host, mhsa.config.input_sequence_size * MAT_SIZE);
	printf("%.2f", sparsity_perc);

	cuda_dealloc_host(mhsa.host);

	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
