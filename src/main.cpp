#include "common.h"
#include "matrix.h"
#include "model.h"

int main(int argc, char* argv[])
{
	DLMC dlmc = { "data/dlmc/transformer/l0_regularization/0.5/" };
	MHSA mhsa = { { 1, 1, 32, dlmc.filepath } };
	try {
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
