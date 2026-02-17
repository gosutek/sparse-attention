#include "utils.h"

#include <iostream>

CSR parse_csr_test_case(const std::filesystem::path& path)
{
	CSR         csr;
	std::string token;
	std::string line;

	std::ifstream stream = { path, std::ios_base::in };
	stream >> csr.rows;
	stream >> csr.cols;
	stream >> csr.nnz;

	csr.row_ptr.resize(csr.rows + 1);
	csr.col_idx.resize(csr.nnz);
	csr.val.resize(csr.nnz);

	for (uint32_t& k : csr.row_ptr) {
		stream >> k;
	}

	for (uint32_t& k : csr.col_idx) {
		stream >> k;
	}

	for (float& k : csr.val) {
		stream >> k;
	}

	return csr;
}

CSC parse_csc_test_case(const std::filesystem::path& path)
{
}

// void generate_token_embeddings(void* dst, size_t size)
// {
// 	float* ptr = reinterpret_cast<float*>(dst);
//
// 	std::random_device                    rd;
// 	std::minstd_rand                      rng(rd());
// 	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
//
// 	for (size_t i = 0; i < size; ++i) {
// 		ptr[i] = uni_real_dist(rng);
// 	}
// }
//
// /*
//  * WARN: This moves the filestream pointer
//  */
// DlmcHeader parse_dlmc_header(std::ifstream& file_stream)
// {
// 	DlmcHeader  res;
// 	std::string token;
// 	std::string header_line;
// 	std::getline(file_stream, header_line);
//
// 	std::istringstream header_stream(header_line);
// 	std::getline(header_stream, token, ',');
// 	res.rows = static_cast<uint32_t>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.cols = static_cast<uint32_t>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.nnz = static_cast<uint32_t>(std::stoul(token));
//
// 	return res;
// }
//
// /*
//  * WARN: This moves the filestream pointer
//  */
// RowMajorHeader parse_row_major_header(std::ifstream& file_stream)
// {
// 	RowMajorHeader res;
// 	std::string    token;
// 	std::string    header_line;
// 	std::getline(file_stream, header_line);
//
// 	std::istringstream header_stream(header_line);
// 	std::getline(header_stream, token, ',');
// 	res.rows = static_cast<uint32_t>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.cols = static_cast<uint32_t>(std::stoul(token));
//
// 	return res;
// }
//
// Csr::Matrix parse_dlmc(void* dst, const std::filesystem::path& filepath)
// {
// 	std::ifstream file_stream(filepath, std::ios_base::in);
//
// 	if (!file_stream) {
// 		// TODO: Remove exceptions
// 		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
// 	}
//
// 	DlmcHeader header = parse_dlmc_header(file_stream);
//
// 	Csr::Matrix csr;
// 	Csr::init(csr, header.rows, header.cols, header.nnz);
// 	Csr::partition(csr, reinterpret_cast<uintptr_t>(dst));
//
// 	std::string line, token;
// 	std::getline(file_stream, line);
// 	std::istringstream row_ptr_stream(line);
// 	for (size_t i = 0; i < csr.row_ptr_count; ++i) {
// 		row_ptr_stream >> token;
// 		csr.row_ptr[i] = static_cast<uint32_t>(std::stoi(token));
// 	}
//
// 	std::getline(file_stream, line);
// 	std::istringstream col_idx_stream(line);
// 	for (size_t i = 0; i < csr.col_idx_count; ++i) {
// 		col_idx_stream >> token;
// 		csr.col_idx[i] = static_cast<uint32_t>(std::stoi(token));
// 	}
//
// 	std::random_device                    rd;
// 	std::minstd_rand                      rng(rd());
// 	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
// 	for (size_t i = 0; i < csr.val_count; ++i) {
// 		csr.val[i] = uni_real_dist(rng);
// 	}
//
// 	return csr;
// }
//
// Csc::Matrix parse_csc_dlmc(void* dst, const std::filesystem::path& filepath)
// {
// 	std::ifstream file_stream(filepath, std::ios_base::in);
//
// 	if (!file_stream) {
// 		// TODO: Remove exceptions
// 		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
// 	}
//
// 	DlmcHeader  header = parse_dlmc_header(file_stream);
// 	Csc::Matrix csc;
// 	Csc::init(csc, header.rows, header.cols, header.nnz);
// 	Csc::partition(csc, reinterpret_cast<uintptr_t>(dst));
//
// 	std::vector<uint32_t> row_ptr_vec(header.rows + 1, 0);
//
// 	std::string line, token;
// 	std::getline(file_stream, line);
// 	std::istringstream row_ptr_stream(line);
// 	for (size_t i = 0; i < header.rows + 1; ++i) {
// 		row_ptr_stream >> token;
// 		row_ptr_vec[i] = static_cast<uint32_t>(std::stoi(token));
// 	}
//
// 	std::vector<uint32_t> col_idx_vec(header.nnz, 0);
//
// 	std::getline(file_stream, line);
// 	std::istringstream col_idx_stream(line);
// 	for (size_t i = 0; i < header.nnz; ++i) {
// 		col_idx_stream >> token;
// 		col_idx_vec[i] = static_cast<uint32_t>(std::stoi(token));
// 	}
//
// 	csr_to_csc(csc, row_ptr_vec, col_idx_vec);
//
// 	std::random_device                    rd;
// 	std::minstd_rand                      rng(rd());
// 	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
// 	for (size_t i = 0; i < csc.val_count; ++i) {
// 		csc.val[i] = uni_real_dist(rng);
// 	}
//
// 	return csc;
// }
