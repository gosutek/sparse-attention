#include "matrix.h"

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
DlmcHeader parse_dlmc_header(std::ifstream& file_stream)
{
	DlmcHeader  res;
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
 * WARN: This moves the filestream pointer
 */
RowMajorHeader parse_row_major_header(std::ifstream& file_stream)
{
	RowMajorHeader res;
	std::string    token;
	std::string    header_line;
	std::getline(file_stream, header_line);

	std::istringstream header_stream(header_line);
	std::getline(header_stream, token, ',');
	res.n_rows = static_cast<size_t>(std::stoi(token));

	std::getline(header_stream, token, ',');
	res.n_cols = static_cast<size_t>(std::stoi(token));

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

[[maybe_unused]] size_t calc_max_nnz_per_col(const Csc::Matrix& csc)
{
	uint32_t res = 0;
	for (size_t i = 0; i < csc.col_ptr_count - 1; ++i) {
		res = std::max(res, csc.col_ptr[i + 1] - csc.col_ptr[i]);
	}
	return res;
}

[[maybe_unused]] std::vector<float> csr_to_row_major(const Csr::Matrix& mat)
{
	std::vector<float> res(mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.rows; ++i) {
		for (size_t j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
			res[i * mat.cols + mat.col_idx[j]] = mat.val[j];
		}
	}
	return res;
}

[[maybe_unused]] std::vector<float> csc_to_col_major(const Csc::Matrix& mat)
{
	std::vector<float> res(mat.rows * mat.cols, 0.0f);

	for (size_t i = 0; i < mat.cols; ++i) {
		for (size_t j = mat.col_ptr[i]; j < mat.col_ptr[i + 1]; ++j) {
			res[i * mat.rows + mat.row_idx[j]] = mat.val[j];
		}
	}
	return res;
}

static void csr_to_csc(Csc::Matrix& mat, const std::vector<uint32_t>& row_ptr_vec, const std::vector<uint32_t>& col_idx_vec)
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
	return nz / static_cast<float>(size);
}

std::vector<float> read_row_major_from_rm(const std::filesystem::path& filepath, size_t size)
{
	if (!std::filesystem::exists(filepath) && !std::filesystem::is_regular_file(filepath)) {
		throw std::runtime_error(filepath.string() + " does not exist\n");
	}
	std::vector<float> res;
	res.reserve(size);

	std::ifstream file_stream(filepath, std::ios_base::in);
	if (!file_stream) {
		throw std::runtime_error("Failed to open file:" + filepath.string());
	}
	float tmp;
	while (file_stream >> tmp) {
		res.push_back(tmp);
	}
	return res;
}

Csr::Matrix parse_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		// TODO: Remove exceptions
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DlmcHeader header = parse_dlmc_header(file_stream);

	Csr::Matrix csr;
	Csr::init(csr, header.rows, header.cols, header.nnz);
	Csr::partition(csr, reinterpret_cast<uintptr_t>(dst));

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < csr.row_ptr_count; ++i) {
		row_ptr_stream >> token;
		csr.row_ptr[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::getline(file_stream, line);
	std::istringstream col_idx_stream(line);
	for (size_t i = 0; i < csr.col_idx_count; ++i) {
		col_idx_stream >> token;
		csr.col_idx[i] = static_cast<uint32_t>(std::stoi(token));
	}

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < csr.val_count; ++i) {
		csr.val[i] = uni_real_dist(rng);
	}

	return csr;
}

Csc::Matrix parse_csc_dlmc(void* dst, const std::filesystem::path& filepath)
{
	std::ifstream file_stream(filepath, std::ios_base::in);

	if (!file_stream) {
		// TODO: Remove exceptions
		throw std::runtime_error("Failed to open file stream for filepath: " + filepath.stem().string());
	}

	DlmcHeader  header = parse_dlmc_header(file_stream);
	Csc::Matrix csc;
	Csc::init(csc, header.rows, header.cols, header.nnz);
	Csc::partition(csc, reinterpret_cast<uintptr_t>(dst));

	std::vector<uint32_t> row_ptr_vec(header.rows + 1, 0);

	std::string line, token;
	std::getline(file_stream, line);
	std::istringstream row_ptr_stream(line);
	for (size_t i = 0; i < header.rows + 1; ++i) {
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

	csr_to_csc(csc, row_ptr_vec, col_idx_vec);

	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);
	for (size_t i = 0; i < csc.val_count; ++i) {
		csc.val[i] = uni_real_dist(rng);
	}

	return csc;
}
