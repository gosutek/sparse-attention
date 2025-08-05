#include "common.h"
#include "matrix.h"
#include "model.h"
#include <cstdlib>

void cuda_dealloc_host(void* ptr);
void run(Input input);

void generate_metadata(std::filesystem::path filepath)
{
}

int main(int argc, char* argv[])
{
	std::string base_data_path = "data/dlmc/transformer/";
	std::string s_pruning_method = "l0_regularization/";
	std::string sparsity = "0.5/";
	int         n_layers = 0;

	if (argc < 2) {
		std::cout << "Running with default data path:\n"
				  << base_data_path
				  << s_pruning_method
				  << sparsity
				  << std::endl;
	} else {
		for (size_t i = 1; i < argc; i += 2) {
			if (argv[i][0] != '-') {
				std::cout << "Flag must be preceded by a dash '-'\n";
			}

			if (argv[i][1] == 'g') {
				generate_metadata({ base_data_path + s_pruning_method + sparsity });
			}
		}
	}

	// TODO: implement passing paths

	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
