#include "matrix.h"
#include "common.h"

#include <random>

void* cuda_malloc_host(size_t size);

/*
 * w_q
 * w_k
 * w_v
 * w_o
 * x
 */

size_t get_mhsa_allocation_size()

	static void parse_dlmc_header(CSRMatrix& mat, std::ifstream& file_stream)
{
	std::string token;
	std::string header_line;
	std::getline(file_stream, header_line);

	std::istringstream header_stream(header_line);
	std::getline(header_stream, token, ',');
	mat.cols = static_cast<size_t>(std::stoi(token));

	std::getline(header_stream, token, ',');
	mat.rows = static_cast<size_t>(std::stoi(token));

	std::getline(header_stream, token, ',');
	mat.nnz = static_cast<size_t>(std::stoi(token));

	mat.row_ptr_size = mat.rows + 1;
	mat.col_idx_size = mat.nnz;
	mat.val_size = mat.nnz;
}

CSRMatrix parse_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		THROW_RUNTIME_ERROR("Error opening file.\n");
	}

	CSRMatrix res;
	parse_dlmc_header(res, file_stream);

	char* ptr = reinterpret_cast<char*>(dst);

	res.row_ptr = reinterpret_cast<uint32_t*>(ptr);
	ptr += res.row_ptr_size * sizeof(uint32_t);

	res.col_idx = reinterpret_cast<uint32_t*>(ptr);
	ptr += res.col_idx_size * sizeof(uint32_t);

	res.val = reinterpret_cast<float*>(ptr);
	ptr += res.val_size * sizeof(float);

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < res.row_ptr_size; ++i) {
		row_ptr_stream >> token;
		res.row_ptr[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::getline(file_stream, line);
	std::istringstream col_idx_stream(line);
	for (size_t i = 0; i < res.col_idx_size; ++i) {
		col_idx_stream >> token;
		res.col_idx[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::getline(file_stream, line);

#if defined(__TEST__)
	for (size_t i = 0; i < res.val_size; ++i) {
		res.val[i] = static_cast<float>(i + 1);
	}
#else
	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < res.val_size; ++i) {
		res.val[i] = uni_real_dist(rng);
	}
#endif

	return res;
}

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
	input.embeddings = q_weights.val + q_weights.val_size;

	for (size_t i = 0; i < embeddings_size; ++i) {
		input.embeddings[i] = uni_real_dist(rng);
	}

	return input;
}

float* csr_to_row_major(CSRMatrix& mat)
{
	float* res = static_cast<float*>(std::malloc(sizeof(float) * mat.rows * mat.cols));
	if (!res) {
		THROW_RUNTIME_ERROR("Failed to allocate");
	}

	std::fill(res, res + mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.rows; ++i) {
		for (size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
			res[i * mat.cols + mat.col_idx[j]] = mat.val[j];
		}
	}
	return res;
}
