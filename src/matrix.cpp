#include "matrix.h"
#include "common.h"
#include "model.h"

#include <cassert>
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

static void generate_token_embeddings(void* dst, size_t input_sequence)
{
	size_t total_size = input_sequence * MAT_SIZE;
	float* ptr = reinterpret_cast<float*>(dst);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	for (size_t i = 0; i < total_size; ++i) {
		ptr[i] = uni_real_dist(rng);
	}
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

static size_t calc_byte_size(const size_t n_rows, const size_t nnz)
{
	size_t b_row_ptr_size = (n_rows + 1) * sizeof(uint32_t);
	size_t b_col_idx_size = nnz * sizeof(uint32_t);
	size_t b_val_size = nnz * sizeof(float);

	return b_row_ptr_size + b_col_idx_size + b_val_size;
}

static Tensor read_tensor(DLMC& dlmc, BodyType bt, AttentionMechanism am, size_t layer)
{
	Tensor tensor;
	tensor.bt = bt;
	tensor.am = am;
	tensor.layer = layer;

	tensor.path = dlmc.base_path + dlmc.pruning_method + dlmc.sparsity;

	if (bt == BodyType::Encoder) {
		tensor.path += "body_encoder_";
	} else {
		tensor.path += "body_decoder_";
	}

	tensor.path += "layer_" + std::to_string(layer) + "_";

	if (am == AttentionMechanism::SelfAttention) {
		tensor.path += "self_attention_multihead_attention_";
	} else {
		tensor.path += "encdec_attention_multihead_attention_";
	}

	for (size_t i = 0; i < dlmc.suffixes.size(); ++i) {
		const auto full_path = tensor.path.string() + dlmc.suffixes[i];
		if (!std::filesystem::exists(full_path) || !std::filesystem::is_regular_file(full_path)) {
			THROW_RUNTIME_ERROR("Tensor component doesn't exist\n");
		}

		std::ifstream file_stream(full_path);
		DLMCHeader    header = parse_dlmc_header(file_stream);
		tensor.b_size += calc_byte_size(header.n_rows, header.nnz);
		tensor.shape[i] = std::move(header);
	}
	return tensor;
}

static CSRMatrix parse_dlmc(void* dst, const std::filesystem::path& filepath)
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

	res.row_ptr_size = header.n_rows + 1;
	res.col_idx_size = header.nnz;
	res.val_size = header.nnz;

	res.row_ptr = reinterpret_cast<uint32_t*>(dst);

	res.col_idx = res.row_ptr + res.row_ptr_size;

	res.val = reinterpret_cast<float*>(res.col_idx + res.col_idx_size);

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
	const std::string& pruning_method,
	const std::string& sparsity,
	AttentionMechanism am)
{
	DLMC dlmc = { base_data_path, pruning_method, sparsity };

	assert(config.n_layers < MAX_N_LAYERS);
	mhsa.b_size = 0;
	for (size_t i = 0; i < config.n_layers; ++i) {
		// WARN: Doing only decoder for now
		// dlmc.enc_self_attention_tensors[i] = read_tensor(dlmc, BodyType::Encoder, am, i);

		dlmc.dec_self_attention_tensors[i] = read_tensor(dlmc, BodyType::Decoder, am, i);
		mhsa.b_size += dlmc.dec_self_attention_tensors[i].b_size;
	}

	size_t b_embeddings_size = config.input_sequence_size * dlmc.dec_self_attention_tensors[0].shape[0].n_rows * sizeof(float);
	mhsa.b_size += b_embeddings_size;

	/*
     * Allocate for
     * w_q (512, 512) float
     * w_k (512, 512) float
     * w_v (512, 512) float
     * w_o (512, 512) float
     * x (input_sequence_size, 512) float
     */

	assert(mhsa.b_size < MAX_ALLOC);
	mhsa.host = cuda_malloc_host(mhsa.b_size);

	if (!mhsa.host) {
		THROW_RUNTIME_ERROR("Failed to allocate page-locked host memory\n");
	}

	weights.x = reinterpret_cast<float*>(mhsa.host);
	generate_token_embeddings(weights.x, config.input_sequence_size);
	try {
		void* block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(mhsa.host) + b_embeddings_size);
		for (size_t i = 0; i < config.n_layers; ++i) {
			void* w_q_ptr = block_start;
			weights.w_q[i] = parse_dlmc(w_q_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[0]);

			size_t b_size = calc_byte_size(dlmc.dec_self_attention_tensors[i].shape[0].n_rows, dlmc.dec_self_attention_tensors[i].shape[0].nnz);
			void*  w_k_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_q_ptr) + b_size);
			weights.w_k[i] = parse_dlmc(w_k_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[1]);

			b_size = calc_byte_size(dlmc.dec_self_attention_tensors[i].shape[1].n_rows, dlmc.dec_self_attention_tensors[i].shape[1].nnz);
			void* w_v_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_k_ptr) + b_size);
			weights.w_v[i] = parse_dlmc(w_v_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[2]);

			b_size = calc_byte_size(dlmc.dec_self_attention_tensors[i].shape[2].n_rows, dlmc.dec_self_attention_tensors[i].shape[2].nnz);
			void* w_o_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_v_ptr) + b_size);
			weights.w_o[i] = parse_dlmc(w_o_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[3]);

			block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(block_start) + dlmc.dec_self_attention_tensors[i].b_size);
		}
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
