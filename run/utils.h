#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

/*
      * +------------------------------------------------------------------------------+
      * |                                 STRUCTS                                      |
      * +------------------------------------------------------------------------------+
*/

struct CSR
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	std::vector<uint32_t> row_ptr{};
	std::vector<uint32_t> col_idx{};
	std::vector<float>    val{};
};

struct CSC
{
	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;

	std::vector<uint32_t> col_ptr;
	std::vector<uint32_t> row_idx;
	std::vector<float>    val;
};

/*
      * +------------------------------------------------------------------------------+
      * |                                 PARSING                                      |
      * +------------------------------------------------------------------------------+
*/

CSR parse_csr_test_case(const std::filesystem::path& path);
CSC parse_csc_test_case(const std::filesystem::path& path);
// void           generate_token_embeddings(void* dst, size_t size);
// DlmcHeader     parse_dlmc_header(std::ifstream& file_stream);
// RowMajorHeader parse_row_major_header(std::ifstream& file_stream);
// Csr::Matrix    parse_dlmc(void* dst, const std::filesystem::path& filepath);
// Csc::Matrix    parse_csc_dlmc(void* dst, const std::filesystem::path& filepath);
