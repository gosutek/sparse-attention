#include "matrix.h"
#include "common.h"
#include "model.h"

#include <fstream>
#include <ios>
#include <random>

void* cuda_malloc_host(size_t size);

/*
 * w_q
 * w_k
 * w_v
 * w_o
 * x
 */

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

CSRMatrix parse_dlmc(void*& dst, const std::filesystem::path& filepath)
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

	dst = reinterpret_cast<void*>(ptr);

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

void read_input(
	MHSA&              mhsa,
	Weights&           weights,
	const std::string& base_data_path,
	const std::string& s_pruning_method,
	const std::string& sparsity,
	const std::string& body,
	const std::string& attention_mechanism,
	const int          layer)
{
	const std::string s_path = base_data_path +
	                           s_pruning_method +
	                           sparsity +
	                           body +
	                           "layer_" + std::to_string(layer) + "_" +
	                           attention_mechanism;

	const std::string q_path = s_path + "q.smtx";
	const std::string k_path = s_path + "k.smtx";
	const std::string v_path = s_path + "v.smtx";
	const std::string o_path = s_path + "outupt_transform.smtx";
	const std::string x_path = base_data_path +
	                           s_pruning_method +
	                           sparsity +
	                           "symbol_modality_33288_512_shared_weights_0_aux.smtx";

	/*
     * Allocate for
     * x (33288, 512) float
     * w_q (512, 512) float
     * w_k (512, 512) float
     * w_v (512, 512) float
     * w_o (512, 512) float
     */

	// 33288 * 512 = token_embeddings_matrix
	// 4 matrices of 512 * 512 maximum size each
	// for 'layer + 1' layers
	mhsa.b_size = sizeof(float) * (33288 * 512) * ((layer + 1) * (4 * 512 * 512));
	void* host = cuda_malloc_host(mhsa.b_size);

	if (!host) {
		THROW_RUNTIME_ERROR("failed to allocate");
	}

	weights.w_q = parse_dlmc(host, q_path);
	weights.w_k = parse_dlmc(host, k_path);
	weights.w_v = parse_dlmc(host, v_path);
	weights.w_o = parse_dlmc(host, o_path);
	weights.x = parse_dlmc(host, x_path);
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
