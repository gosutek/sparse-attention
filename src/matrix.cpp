#include "matrix.h"
#include "handle.h"

#include <cassert>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

void* cuda_malloc_host(size_t size);
void  cuda_dealloc_host(void* ptr);

/*
 * w_q
 * w_k
 * w_v
 * w_o
 * x
 */

void generate_token_embeddings(void* dst, size_t size)
{
	float* ptr = reinterpret_cast<float*>(dst);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	for (size_t i = 0; i < size; ++i) {
		ptr[i] = uni_real_dist(rng);
	}
}

/*
 * WARN: This moves the filestream pointer
 */
DLMCHeader parse_dlmc_header(std::ifstream& file_stream)
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

/*
 * Calculates the size of a CSR or CSC matrix in bytes for float values
 * Accounts for non-square matrices
 * n: main dimension's size (cols for CSC, rows for CSR)
 */
size_t calc_sparse_b_size(const size_t n, const size_t nnz)
{
	size_t b_ptr_size = (n + 1) * sizeof(uint32_t);
	size_t b_idx_size = nnz * sizeof(uint32_t);
	size_t b_val_size = nnz * sizeof(float);

	return b_ptr_size + b_idx_size + b_val_size;
}

size_t calc_max_nnz_per_col(const CSC& csc)
{
	uint32_t res = 0;
	for (size_t i = 0; i < csc.col_ptr_size - 1; ++i) {
		res = std::max(res, csc.col_ptr[i + 1] - csc.col_ptr[i]);
	}
	return res;
}

std::vector<float> csr_to_row_major(const CSR& mat)
{
	std::vector<float> res(mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.rows; ++i) {
		for (size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
			res[i * mat.cols + mat.col_idx[j]] = mat.val[j];
		}
	}
	return res;
}

std::vector<float> csc_to_col_major(const CSC& mat)
{
	std::vector<float> res(mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.cols; ++i) {
		for (size_t j = mat.col_ptr[i]; j < mat.col_ptr[i + 1]; ++j) {
			res[i * mat.rows + mat.row_idx[j]] = mat.val[j];
		}
	}
	return res;
}

static void csr_to_csc(CSC& mat, const std::vector<uint32_t>& row_ptr_vec, const std::vector<uint32_t>& col_idx_vec)
{
	std::vector<uint32_t> col_count(mat.cols, 0);
	for (size_t i = 0; i < mat.nnz; ++i) {
		col_count[col_idx_vec[i]]++;
	}

	mat.col_ptr[0] = 0;
	for (size_t col = 0; col < mat.cols; ++col) {
		mat.col_ptr[col + 1] = mat.col_ptr[col] + col_count[col];
	}

	std::vector<uint32_t> cur_pos(mat.cols);
	for (size_t col = 0; col < mat.cols; ++col) {
		cur_pos[col] = mat.col_ptr[col];
	}

	for (uint32_t row = 0; row < mat.rows; ++row) {
		for (size_t i = row_ptr_vec[row]; i < row_ptr_vec[row + 1]; ++i) {
			uint32_t col = col_idx_vec[i];
			uint32_t dest_pos = cur_pos[col]++;
			mat.row_idx[dest_pos] = row;
		}
	}
}

float measure_sparsity(void* s, size_t size)
{
	float* ptr = reinterpret_cast<float*>(s);
	float  nz = .0f;
	for (size_t i = 0; i < size; i++) {
		if (ptr[i] == 0)
			nz++;
	}
	return nz / size;
}

std::string construct_path(const std::filesystem::path base_path, const BodyType bt, const AttentionMechanism am, const size_t layer)
{
	std::string path = base_path;
	if (bt == BodyType::Encoder) {
		path += "body_encoder_";
	} else {
		path += "body_decoder_";
	}
	path += "layer_" + std::to_string(layer) + "_";

	if (am == AttentionMechanism::SelfAttention) {
		path += "self_attention_multihead_attention_";
	} else {
		path += "encdec_attention_multihead_attention_";
	}
	return path;
}

Tensor read_tensor(const DLMC& dlmc, const BodyType bt, const AttentionMechanism am, const size_t layer, const SparseMatrixType sparse_matrix_type)
{
	Tensor tensor;

	std::string data_dir_string = construct_path(dlmc.base_path + dlmc.pruning_method + dlmc.sparsity, bt, am, layer);

	for (size_t i = 0; i < dlmc.suffixes.size(); ++i) {
		const auto full_path = tensor.path.string() + dlmc.suffixes[i];
		if (!std::filesystem::exists(full_path) || !std::filesystem::is_regular_file(full_path)) {
			throw std::runtime_error("Tensor component doesn't exist: " + full_path);
		}

		std::ifstream file_stream(full_path);
		DLMCHeader    header = parse_dlmc_header(file_stream);

		if (sparse_matrix_type == SparseMatrixType::CSC) {
			tensor.b_size += calc_sparse_b_size(header.n_cols, header.nnz);
		} else {
			tensor.b_size += calc_sparse_b_size(header.n_rows, header.nnz);
		}
		tensor.shape[i] = std::move(header);
	}
	return tensor;
}

static CSR read_mask(const DLMC& dlmc, const size_t sequence_size, const size_t band_size_ratio, const size_t sparsity)
{
	const std::filesystem::path path = std::format("{}{}{}m_{}_0{}_{}.smtx",
		dlmc.base_path, dlmc.pruning_method, dlmc.sparsity,
		sequence_size, band_size_ratio, sparsity);

	if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
		throw std::runtime_error("Mask file doesn't exist: " + path.stem().string());
	}

	std::ifstream file_stream(path);
	DLMCHeader    header = parse_dlmc_header(file_stream);

	return { header.n_rows, header.n_cols, header.nnz };
}

CSR parse_csr_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DLMCHeader header = parse_dlmc_header(file_stream);
	CSR        res = { header.n_rows, header.n_cols, header.nnz };

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

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < res.val_size; ++i) {
		res.val[i] = uni_real_dist(rng);
	}

	return res;
}

CSC parse_csc_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DLMCHeader header = parse_dlmc_header(file_stream);
	CSC        res(header.n_rows, header.n_cols, header.nnz);

	uintptr_t ptr = reinterpret_cast<uintptr_t>(dst);
	res.col_ptr = reinterpret_cast<uint32_t*>(ptr);

	size_t b_size = res.col_ptr_size * sizeof(uint32_t);
	ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
	res.row_idx = reinterpret_cast<uint32_t*>(ptr);

	b_size = res.row_idx_size * sizeof(uint32_t);
	ptr += b_size + calc_padding_bytes(b_size, ALIGNMENT_BYTES);
	res.val = reinterpret_cast<float*>(ptr);

	std::vector<uint32_t> row_ptr_vec(header.n_rows + 1, 0);

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < header.n_rows + 1; ++i) {
		row_ptr_stream >> token;
		row_ptr_vec[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::vector<uint32_t> col_idx_vec(header.nnz, 0);

	std::getline(file_stream, line);
	std::istringstream col_idx_stream(line);
	for (size_t i = 0; i < header.nnz; ++i) {
		col_idx_stream >> token;
		col_idx_vec[i] = static_cast<uint32_t>(std::stoi(token));
	}

	csr_to_csc(res, row_ptr_vec, col_idx_vec);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < res.val_size; ++i) {
		res.val[i] = uni_real_dist(rng);
	}

	return res;
}

// static CSC read_mat(void* dst, const BodyType bt, const AttentionMechanism am, const size_t layer)
// {
// 	const std::filesystem::path base_path = "data/dlmc/transformer/l0_regularization/0.5/";
//
// 	std::string data_dir_string = construct_path(base_path, bt, am, layer);
// 	data_dir_string += "_q.smtx";
//
// 	std::filesystem::path path = { data_dir_string };
//
// 	if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
// 		throw std::runtime_error("Matrix doesn't exist: " + path.stem().string());
// 	}
// 	std::ifstream file_stream(path);
//
// 	CSC mat = parse_csc_dlmc(dst, path);
//
// 	return mat;
// }

// void mhsa_load_host_csr(
// 	MHSA<CSR, CSR> mhsa,
// 	const Config&  config,
// 	DLMC&          dlmc,
// 	Weights<CSR>&  weights)
// {
// 	assert(config.n_layers < MAX_N_LAYERS);
// 	mhsa.b_size = 0;
// 	for (size_t i = 0; i < config.n_layers; ++i) {
// 		// WARN: Doing only decoder for now
// 		// dlmc.enc_self_attention_tensors[i] = read_tensor(dlmc, BodyType::Encoder, am, i);
//
// 		dlmc.dec_self_attention_tensors[i] = read_tensor(dlmc, dlmc.bt, dlmc.am, i, SparseMatrixType::CSR);
// 		mhsa.b_size += dlmc.dec_self_attention_tensors[i].b_size;
// 	}
//
// 	size_t b_embeddings_size = config.input_sequence_size * dlmc.dec_self_attention_tensors[0].shape[0].n_rows * sizeof(float);
// 	mhsa.b_size += b_embeddings_size;
//
// 	mhsa.mask = read_mask(dlmc, mhsa.config.input_sequence_size, 2, 95);
// 	mhsa.b_size += mhsa.mask.b_size;
//
// 	/*
//      * Allocate for
//      * x (input_sequence_size, 512) float
//      * w_q (512, 512) float
//      * w_k (512, 512) float
//      * w_v (512, 512) float
//      * w_o (512, 512) float
//      * mask
//      */
//
// 	assert(mhsa.b_size < MAX_ALLOC);
// 	mhsa.host = cuda_malloc_host(mhsa.b_size);
//
// 	if (!mhsa.host) {
// 		throw std::runtime_error("Failed to allocate page-locked host memory.");
// 	}
//
// 	mhsa.x = reinterpret_cast<float*>(mhsa.host);
// 	generate_token_embeddings(mhsa.x, config.input_sequence_size * MAT_SIZE);
// 	try {
// 		void* block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(mhsa.host) + b_embeddings_size);
// 		for (size_t i = 0; i < config.n_layers; ++i) {
// 			void* w_q_ptr = block_start;
// 			weights.w_q[i] = parse_csr_dlmc(w_q_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[0]);
//
// 			size_t b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[0].n_rows, dlmc.dec_self_attention_tensors[i].shape[0].nnz);
// 			void*  w_k_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_q_ptr) + b_size);
// 			weights.w_k[i] = parse_csr_dlmc(w_k_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[1]);
//
// 			b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[1].n_rows, dlmc.dec_self_attention_tensors[i].shape[1].nnz);
// 			void* w_v_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_k_ptr) + b_size);
// 			weights.w_v[i] = parse_csr_dlmc(w_v_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[2]);
//
// 			b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[2].n_rows, dlmc.dec_self_attention_tensors[i].shape[2].nnz);
// 			void* w_o_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_v_ptr) + b_size);
// 			weights.w_o[i] = parse_csr_dlmc(w_o_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[3]);
//
// 			block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(block_start) + dlmc.dec_self_attention_tensors[i].b_size);
// 		}
//
// 	} catch (const std::exception& e) {
// 		cuda_dealloc_host(mhsa.host);
// 		throw;
// 	}
// }
//
// void mhsa_load_host_csc(
// 	MHSA<CSC, CSR>& mhsa,
// 	const Config&   config,
// 	DLMC&           dlmc,
// 	Weights<CSC>&   weights)
// {
// 	assert(mhsa.config.n_layers < MAX_N_LAYERS);
// 	for (size_t i = 0; i < mhsa.config.n_layers; ++i) {
// 		// WARN: Doing only decoder for now
// 		// dlmc.enc_self_attention_tensors[i] = read_tensor(dlmc, BodyType::Encoder, am, i);
//
// 		dlmc.dec_self_attention_tensors[i] = read_tensor(dlmc, dlmc.bt, dlmc.am, i, SparseMatrixType::CSC);
// 		mhsa.b_size += dlmc.dec_self_attention_tensors[i].b_size;
// 	}
//
// 	size_t b_embeddings_size = mhsa.config.input_sequence_size * dlmc.dec_self_attention_tensors[0].shape[0].n_rows * sizeof(float);
// 	mhsa.b_size += b_embeddings_size;
//
// 	mhsa.mask = read_mask(dlmc, mhsa.config.input_sequence_size, 2, 95);
// 	mhsa.b_size += mhsa.mask.b_size;
//
// 	/*
//      * Allocate for
//      * w_q (512, 512) float
//      * w_k (512, 512) float
//      * w_v (512, 512) float
//      * w_o (512, 512) float
//      * x (input_sequence_size, 512) float
//      */
//
// 	assert(mhsa.b_size < MAX_ALLOC);
// 	mhsa.host = cuda_malloc_host(mhsa.b_size);
//
// 	if (!mhsa.host) {
// 		throw std::runtime_error("Failed to allocate page-locked host memory.");
// 	}
//
// 	try {
// 		mhsa.x = reinterpret_cast<float*>(mhsa.host);
// 		generate_token_embeddings(mhsa.x, config.input_sequence_size * MAT_SIZE);
// 		void* block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(mhsa.host) + b_embeddings_size);
// 		for (size_t i = 0; i < config.n_layers; ++i) {
// 			void* w_q_ptr = block_start;
// 			weights.w_q[i] = parse_csc_dlmc(w_q_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[0]);
//
// 			size_t b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[0].n_cols, dlmc.dec_self_attention_tensors[i].shape[0].nnz);
// 			void*  w_k_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_q_ptr) + b_size);
// 			weights.w_k[i] = parse_csc_dlmc(w_k_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[1]);
//
// 			b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[1].n_cols, dlmc.dec_self_attention_tensors[i].shape[1].nnz);
// 			void* w_v_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_k_ptr) + b_size);
// 			weights.w_v[i] = parse_csc_dlmc(w_v_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[2]);
//
// 			b_size = calc_sparse_b_size(dlmc.dec_self_attention_tensors[i].shape[2].n_cols, dlmc.dec_self_attention_tensors[i].shape[2].nnz);
// 			void* w_o_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(w_v_ptr) + b_size);
// 			weights.w_o[i] = parse_csc_dlmc(w_o_ptr, dlmc.dec_self_attention_tensors[i].path.string() + dlmc.suffixes[3]);
//
// 			block_start = reinterpret_cast<void*>(reinterpret_cast<char*>(block_start) + dlmc.dec_self_attention_tensors[i].b_size);
// 		}
//
// 		// TODO: fix the way path is passed
// 		mhsa.mask = parse_csr_dlmc(reinterpret_cast<void*>(block_start), dlmc.base_path + dlmc.pruning_method + dlmc.sparsity + "m_32_02_95.smtx");
// 	} catch (const std::exception& e) {
// 		cuda_dealloc_host(mhsa.host);
// 		throw;
// 	}
// }
