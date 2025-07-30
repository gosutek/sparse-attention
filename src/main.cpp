#include "common.h"
#include "matrix.h"

void cuda_dealloc_host(void* ptr);
void run(Input input);

int main()
{
	const auto data_path = std::filesystem::current_path() / DATA_DIRECTORY / "dlmc/transformer/l0_regularization/0.5/body_decoder_layer_0_self_attention_multihead_attention_q.smtx";

	try {
		Input input = read_input(data_path);
		run(input);

		std::cout << reinterpret_cast<float*>(input.data)[0] << "\n";

		cuda_dealloc_host(input.data);

	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
