#include "utils.h"

#include <fstream>

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

CSC parse_csc_dlmc(const std::filesystem::path& filepath)
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

	CSC csc = {
		.rows = csr.rows,
		.cols = csr.cols,
		.nnz = csr.nnz,
		.col_ptr = std::vector<u32>(csr.cols + 1, 0),
		.row_idx = std::vector<u32>(csr.nnz),
		.val = std::vector<float>(csr.nnz),
	};

	for (u32 i = 0; i < csr.nnz; ++i) {
		csc.col_ptr[csr.col_idx[i] + 1]++;
	}

	for (u32 i = 1; i < csc.cols + 1; ++i) {
		csc.col_ptr[i] += csc.col_ptr[i - 1];
	}

	std::vector<u32> work_buf = csc.col_ptr;  // deep copy
	for (u32 row = 0; row < csr.rows; ++row) {
		for (u32 i = csr.row_ptr[row]; i < csr.row_ptr[row + 1]; ++i) {
			const u32 col = csr.col_idx[i];
			const u32 dest_pos = work_buf[col]++;
			csc.row_idx[dest_pos] = row;
			csc.val[dest_pos] = csr.val[i];
		}
	}

	return csc;
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
