#include "matrix.h"
#include "common.h"

#include <random>

void* cuda_malloc_host(size_t size);

Input read_input(const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		THROW_RUNTIME_ERROR("Error opening file.\n");
	}

	Input      input;
	CSRMatrix& q_weights = input.weights[0];

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	std::string header_line{}, line{}, token{};

	std::getline(file_stream, header_line);
	std::istringstream header_stream(header_line);
	std::getline(header_stream, token, ',');
	q_weights.cols = static_cast<uint32_t>(std::stoi(token));
	std::getline(header_stream, token, ',');
	q_weights.rows = static_cast<uint32_t>(std::stoi(token));
	std::getline(header_stream, token, ',');
	q_weights.nnz = static_cast<uint32_t>(std::stoi(token));

	q_weights.row_ptr_size = q_weights.rows + 1;
	q_weights.col_idx_size = q_weights.nnz;
	q_weights.val_size = q_weights.nnz;
	uint32_t embeddings_size = q_weights.rows * q_weights.rows;

	input.b_size =
		q_weights.row_ptr_size * sizeof(uint32_t) +
		q_weights.col_idx_size * sizeof(uint32_t) +
		q_weights.val_size * sizeof(float) +
		embeddings_size * sizeof(float);

	// TODO: This should allocate for the result aswell
	input.data = cuda_malloc_host(input.b_size);
	if (!input.data) {
		THROW_RUNTIME_ERROR("failed to allocate");
	}
	q_weights.row_ptr = reinterpret_cast<uint32_t*>(input.data);
	q_weights.col_idx = q_weights.row_ptr + q_weights.row_ptr_size;
	q_weights.val = reinterpret_cast<float*>(q_weights.col_idx + q_weights.col_idx_size);
	input.embeddings = q_weights.val + q_weights.val_size;

	std::getline(file_stream, line);
	std::istringstream row_ptr_line(line);

	uint32_t idx = 0;
	while (row_ptr_line >> token) {
		q_weights.row_ptr[idx++] = static_cast<uint32_t>(std::stoi(token));
	}

	std::getline(file_stream, line);
	std::istringstream col_idx_line(line);

	idx = 0;
	while (col_idx_line >> token) {
		q_weights.col_idx[idx] = static_cast<uint32_t>(std::stoi(token));
		q_weights.val[idx++] = uni_real_dist(rng);
	}

	for (size_t i = 0; i < embeddings_size; ++i) {
		input.embeddings[i] = uni_real_dist(rng);
	}

	return input;
}
