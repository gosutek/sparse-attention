#include "common.h"

#include "../extern/mmio.h"

#include <filesystem>
#include <fstream>
#include <random>

struct COOElement
{
	uint32_t row, col;

	float val;
};

struct COOMatrix
{
	uint32_t rows, cols, nnz;

	std::vector<COOElement> elements;
};

struct SerializationInput
{
	uint32_t           rows{};
	uint32_t           cols{};
	std::vector<float> sparse_elements;
	std::vector<float> dense_elements;
};

/*
 * Generates a dense matrix in column-major format
 * of size rows * cols filled with random values
 */
// TODO: Made static after testing
static std::vector<float> generate_dense(size_t size)
{
	std::random_device                    rd;
	std::minstd_rand                      rng(rd());
	std::uniform_real_distribution<float> uni_real_dist(0.0f, 1.0f);

	std::vector<float> dense_values;
	std::cout << "Generating Dense Matrix..." << std::flush;
	dense_values.reserve(size);
	for (size_t i = 0; i < size; ++i) {
		float half_random_value = uni_real_dist(rng);
		dense_values.push_back(half_random_value);
	}
	std::cout << "Done!" << std::endl;
	;

	return dense_values;
}

COOMatrix read_mtx(const std::filesystem::path& filepath)
{
	FILE* f = fopen(filepath.c_str(), "r");
	if (!f)
		THROW_RUNTIME_ERROR("Failed to open file ~ " + filepath.filename().string());

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Error reading mtx banner");
	}

	if (!mm_is_sparse(matcode)) {
		fclose(f);
		THROW_RUNTIME_ERROR("Matrix is not sparse");
	}

	int rows, cols, nnz;
	if (mm_read_mtx_crd_size(f, &rows, &cols, &nnz) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Failed to read matrix size");
	}

	std::vector<COOElement> elements;
	elements.reserve(static_cast<size_t>(nnz));

	for (int i = 0; i < nnz; ++i) {
		COOElement e;
		if (fscanf(f, "%u %u %f\n", &e.row, &e.col, &e.val) != 3) {
			fclose(f);
			THROW_RUNTIME_ERROR("Error reading element ~ " + std::to_string(i));
		}
		e.row--;
		e.col--;
		elements.push_back(e);
	}
	fclose(f);
	return { static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), static_cast<uint32_t>(nnz), std::move(elements) };
}

void print_matrix_specs(const std::filesystem::path& filepath)
{
	FILE* f = fopen(filepath.c_str(), "r");
	int   rows = 0;
	int   cols = 0;
	int   nnz = 0;

	if (!f)
		THROW_RUNTIME_ERROR("Failed to open file ~ " + filepath.filename().string());

	MM_typecode matcode;
	if (mm_read_banner(f, &matcode) != 0) {
		fclose(f);
		THROW_RUNTIME_ERROR("Error reading mtx banner");
	}

	std::cout << filepath.filename() << "\n";
	for (int i = 0; i < 4; ++i) {
		std::cout << matcode[i];
	}
	std::cout << "\n";

	if (mm_is_sparse(matcode)) {
		mm_read_mtx_crd_size(f, &rows, &cols, &nnz);
		std::cout << "Sparse with " << rows << " rows and " << cols << " cols and nnz " << nnz << "\n";
		std::cout << "Data type " << matcode[2] << "\n";
	} else {
		mm_read_mtx_array_size(f, &rows, &cols);
		std::cout << "Dense with " << rows << " rows and " << cols << " cols.\n";
	}
}

static std::filesystem::path replace_extension(const std::filesystem::path& path, const char* ext)
{
	return path.parent_path() / path.filename().replace_extension(ext);
}

/*
 * Checks if [path] has a .mtx extension 
 * and
 * if the same filename exists as a .csr
 */
static bool requires_conversion(const std::filesystem::path& path)
{
	// TODO: Add a check for .bsr when it's implemented
	// Add a check for KBSR when it's implemented
	return path.extension().string() == ".mtx" &&
	       !(std::filesystem::exists(replace_extension(path, ".spmm")));
}

static std::vector<float> coo_to_row_major(const COOMatrix& mtx)
{
	std::vector<float> re;
	re.resize(mtx.rows * mtx.cols, 0);

	for (const COOElement& e : mtx.elements) {
		re[e.row * mtx.cols + e.col] = e.val;
	}

	return re;
}

static void serialize(const std::filesystem::path& path, const SerializationInput& data)
{
	std::ofstream ofs(path, std::ios::binary);

	ofs.write(reinterpret_cast<const char*>(&data.rows), sizeof(data.rows));
	ofs.write(reinterpret_cast<const char*>(&data.cols), sizeof(data.cols));
	ofs.write(reinterpret_cast<const char*>(data.sparse_elements.data()), data.sparse_elements.size() * sizeof(float));
	ofs.write(reinterpret_cast<const char*>(data.dense_elements.data()), data.dense_elements.size() * sizeof(float));

	ofs.close();
}

static void batch_convert(const std::filesystem::directory_iterator& data_dir)
{
	for (const auto& filepath : data_dir) {
		if (filepath.is_regular_file() && requires_conversion(filepath.path())) {
			COOMatrix          mtx = read_mtx(filepath);
			std::vector<float> sparse_vec = coo_to_row_major(mtx);
			std::vector<float> dense_vec = generate_dense(mtx.rows * mtx.cols);

			std::filesystem::path output_filepath = replace_extension(filepath, ".spmm");
			serialize(output_filepath, { mtx.rows, mtx.cols, std::move(sparse_vec), std::move(dense_vec) });
		}
	}
}

static void unit_test_serialization(const std::filesystem::path& filepath)
{
	COOMatrix unit_coo = {
		4, 4, 4,
		{ { 1, 1, 2.0f },
			{ 2,
				3,
				5.0f },
			{ 2,
				1,
				6.0f },
			{ 0, 3, 69.0f } }
	};
	std::vector<float> sparse = coo_to_row_major(unit_coo);
	std::vector<float> dense = {
		1.0f, 2.0f, 3.0f, 4.0f,
		1.0f, 2.0f, 3.0f, 4.0f,
		1.0f, 2.0f, 3.0f, 4.0f,
		1.0f, 2.0f, 3.0f, 4.0f
	};

	const SerializationInput input = {
		4, 4,
		std::move(sparse),
		std::move(dense)
	};

	uint32_t in_rows{};
	uint32_t in_cols{};
	sparse = {};
	dense = {};

	serialize(filepath, input);

	std::ifstream ifs(filepath, std::ios::binary);

	ifs.read(reinterpret_cast<char*>(&in_rows), sizeof(in_rows));
	printf("Read %lu bytes, rows are equal to %d\n", sizeof(in_rows), in_rows);
	ifs.read(reinterpret_cast<char*>(&in_cols), sizeof(in_cols));
	printf("Read %lu bytes, cols are equal to %d\n", sizeof(in_cols), in_cols);
	printf("Reserving %lu bytes of memory for the vector of sparse elements (%u)\n", in_rows * in_cols * sizeof(float), in_rows * in_cols);
	sparse.resize(in_rows * in_cols);
	printf("Reserving %lu bytes of memory for the vector of dense elements (%u)\n", in_rows * in_cols * sizeof(float), in_rows * in_cols);
	dense.resize(in_rows * in_cols);

	printf("Reading %lu bytes, or %u elements of the vector of sparse elements\n", in_rows * in_cols * sizeof(float), in_rows * in_cols);
	ifs.read(reinterpret_cast<char*>(sparse.data()), in_rows * in_cols * sizeof(float));

	printf("Reading %lu bytes, or %u elements of the vector of dense elements\n", in_rows * in_cols * sizeof(float), in_rows * in_cols);
	ifs.read(reinterpret_cast<char*>(dense.data()), in_rows * in_cols * sizeof(float));

	std::cout << "Sparse matrix\n"
			  << sparse << "\n";
	std::cout << "Dense matrix\n"
			  << dense << "\n";
}

static void unit_test_coo_to_row_major()
{
	COOMatrix unit_coo = {
		4, 4, 4,
		{ { 1, 1, 2 },
			{ 2,
				3,
				5 },
			{ 2,
				1,
				6 },
			{ 0, 3, 69 } }
	};
	const auto& re = coo_to_row_major(unit_coo);
	for (const auto& e : re) {
		std::cout << e << std::endl;
	}
}

int main()
{
	const auto data_path = std::filesystem::current_path() / DATA_DIRECTORY / "fv1/";
	const auto data_dir = std::filesystem::directory_iterator(std::move(data_path));
	try {
		batch_convert(data_dir);
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}
}
