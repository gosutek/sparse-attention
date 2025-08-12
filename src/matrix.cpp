#include "matrix.h"
#include "common.h"
#include "model.h"

#include <random>

void* cuda_malloc_host(size_t size);
void  cuda_dealloc_host(void* ptr);

/*
 * w_q
 * w_k
 * w_v
 * w_o
 * x
 */

static float* generate_token_embeddings(void*& dst, size_t input_sequence)
{
	size_t total_size = input_sequence * MAT_SIZE;
	float* ptr = reinterpret_cast<float*>(dst);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	for (size_t i = 0; i < total_size; ++i) {
		ptr[i] = uni_real_dist(rng);
	}

	// Move the global ptr of the block to the end of the embeddings
	dst = reinterpret_cast<void*>(reinterpret_cast<char*>(dst) + (total_size * sizeof(float)));
	return ptr;
}

/*
 * WARN: This moves the filestream pointer
 */
static DLMCHeader parse_dlmc_header(std::ifstream& file_stream)
{
	DLMCHeader  res;
	std::string token;
	std::string header_line;
	std::getline(file_stream, header_line);

	std::istringstream header_stream(header_line);
	std::getline(header_stream, token, ',');
	res.n_rows = static_cast<size_t>(std::stoi(token));

	std::getline(header_stream, token, ',');
	res.n_cols = static_cast<size_t>(std::stoi(token));

	std::getline(header_stream, token, ',');
	res.nnz = static_cast<size_t>(std::stoi(token));

	return res;
}





static size_t get_byte_size(const size_t n_rows, const size_t nnz)
{
	size_t b_row_ptr_size = (n_rows + 1) * sizeof(uint32_t);
	size_t b_col_idx_size = nnz * sizeof(uint32_t);
	size_t b_val_size = nnz * sizeof(float);

	return b_row_ptr_size + b_col_idx_size + b_val_size;
}


static CSRMatrix parse_dlmc(void*& dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		THROW_RUNTIME_ERROR(filepath.string());
	}

	CSRMatrix  res;
	DLMCHeader header = parse_dlmc_header(file_stream);
	res.rows = header.n_rows;
	res.cols = header.n_cols;
	res.nnz = header.nnz;

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
	Config&            config,
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
	const std::string o_path = s_path + "output_transform.smtx";

	/*
     * Allocate for
     * w_q (512, 512) float
     * w_k (512, 512) float
     * w_v (512, 512) float
     * w_o (512, 512) float
     * x (512, 512) float
     */

	mhsa.host = cuda_malloc_host(MAX_ALLOC);

	if (!mhsa.host) {
		THROW_RUNTIME_ERROR("failed to allocate");
	}

	void* ptr = mhsa.host;

	try {
		weights.w_q = parse_dlmc(ptr, q_path);

		weights.w_k = parse_dlmc(ptr, k_path);

		weights.w_v = parse_dlmc(ptr, v_path);

		weights.w_o = parse_dlmc(ptr, o_path);

		// TODO: Pass the main host ptr and get a copy of a ptr that start at the embeddings table
		weights.x = generate_token_embeddings(ptr, config.input_sequence_size);

	} catch (const std::exception& e) {
		cuda_dealloc_host(mhsa.host);
		throw;
	}
}

std::vector<float> csr_to_row_major(const CSRMatrix& mat)
{
	std::vector<float> res(mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.rows; ++i) {
		for (size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
			res[i * mat.cols + mat.col_idx[j]] = mat.val[j];
		}
	}
	return res;
}
