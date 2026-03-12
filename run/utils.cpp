#include "utils.h"

Dense parse_dn_test_case(const std::filesystem::path& path)
{
	Dense       rm;
	std::string token;
	std::string line;

	std::ifstream stream = { path, std::ios_base::in };
	stream >> rm.rows;
	stream >> rm.cols;

	rm.val.resize(rm.rows * rm.cols);

	for (f32& k : rm.val) {
		stream >> k;
	}
	return rm;
}

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

	for (u32& k : csr.row_ptr) {
		stream >> k;
	}

	for (u32& k : csr.col_idx) {
		stream >> k;
	}

	for (f32& k : csr.val) {
		stream >> k;
	}

	return csr;
}

CSR parse_csr_dlmc(const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		// TODO: Remove exceptions
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	CSR csr;

	std::string token;
	std::string line;
	std::getline(file_stream, line);

	std::istringstream line_stream(line);
	std::getline(line_stream, token, ',');

	csr.rows = static_cast<u32>(std::stoul(token));

	std::getline(line_stream, token, ',');
	csr.cols = static_cast<u32>(std::stoul(token));

	std::getline(line_stream, token, ',');
	csr.nnz = static_cast<u32>(std::stoul(token));

	csr.row_ptr.resize(csr.rows + 1);
	csr.col_idx.resize(csr.nnz);
	csr.val.resize(csr.nnz);

	for (u32& k : csr.row_ptr) {
		file_stream >> k;
	}

	for (u32& k : csr.col_idx) {
		file_stream >> k;
	}

	for (f32& k : csr.val) {
		file_stream >> k;
	}

	gen_synth_weights_vec<f32>(csr.val, csr.nnz);

	return csr;
}

CSC parse_csc_test_case(const std::filesystem::path& path)
{
	CSC         csc;
	std::string token;
	std::string line;

	std::ifstream stream = { path, std::ios_base::in };
	stream >> csc.rows;
	stream >> csc.cols;
	stream >> csc.nnz;

	csc.col_ptr.resize(csc.cols + 1);
	csc.row_idx.resize(csc.nnz);
	csc.val.resize(csc.nnz);

	for (u32& k : csc.col_ptr) {
		stream >> k;
	}

	for (u32& k : csc.row_idx) {
		stream >> k;
	}

	for (f32& k : csc.val) {
		stream >> k;
	}

	return csc;
}

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
// 	res.rows = static_cast<u32>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.cols = static_cast<u32>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.nnz = static_cast<u32>(std::stoul(token));
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
// 	res.rows = static_cast<u32>(std::stoul(token));
//
// 	std::getline(header_stream, token, ',');
// 	res.cols = static_cast<u32>(std::stoul(token));
//
// 	return res;
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
// 	std::vector<u32> row_ptr_vec(header.rows + 1, 0);
//
// 	std::string line, token;
// 	std::getline(file_stream, line);
// 	std::istringstream row_ptr_stream(line);
// 	for (size_t i = 0; i < header.rows + 1; ++i) {
// 		row_ptr_stream >> token;
// 		row_ptr_vec[i] = static_cast<u32>(std::stoi(token));
// 	}
//
// 	std::vector<u32> col_idx_vec(header.nnz, 0);
//
// 	std::getline(file_stream, line);
// 	std::istringstream col_idx_stream(line);
// 	for (size_t i = 0; i < header.nnz; ++i) {
// 		col_idx_stream >> token;
// 		col_idx_vec[i] = static_cast<u32>(std::stoi(token));
// 	}
//
// 	csr_to_csc(csc, row_ptr_vec, col_idx_vec);
//
// 	std::random_device                    rd;
// 	std::minstd_rand                      rng(rd());
// 	std::uniform_real_distribution<f32> uni_real_dist(0.0f, 1.0f);
// 	for (size_t i = 0; i < csc.val_count; ++i) {
// 		csc.val[i] = uni_real_dist(rng);
// 	}
//
// 	return csc;
// }
