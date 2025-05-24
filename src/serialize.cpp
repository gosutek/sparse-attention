#include "common.h"

#include "../extern/mmio.h"

#include <cassert>
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
	std::vector<float> sparse_elements{};
	std::vector<float> dense_elements{};
	uint32_t           rows{};
	uint32_t           cols{};
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

static size_t calculate_padding(size_t size)
{
	size_t remainder = size % ALIGNMENT;
	if (ALIGNMENT == 0 || remainder == 0) {
		return 0;
	} else {
		return ALIGNMENT - remainder;
	}
}

static void deserialize(const std::filesystem::path& path)
{
	// TODO
}

static void serialize(const std::filesystem::path& path, const SerializationInput& data)
{
	std::ofstream ofs(path, std::ios::binary);

	size_t padding = 0;
	size_t chunk_size = 0;

	ofs.write(reinterpret_cast<const char*>(&data.rows), sizeof(data.rows));
	ofs.write(reinterpret_cast<const char*>(&data.cols), sizeof(data.cols));

	chunk_size = sizeof(data.rows) + sizeof(data.cols);
	padding = calculate_padding(chunk_size);
	printf("Metadata: Size equal to %lu, I need to pad %lu bytes\n", chunk_size, padding);
	ofs.write(reinterpret_cast<const char*>(&padding), sizeof(padding));

	ofs.write(reinterpret_cast<const char*>(data.sparse_elements.data()), data.sparse_elements.size() * sizeof(float));
	chunk_size = data.sparse_elements.size() * sizeof(float);
	padding = calculate_padding(chunk_size);
	printf("Sparse Matrix: Size equal to %lu, I need to pad %lu bytes\n", chunk_size, padding);
	ofs.write(reinterpret_cast<const char*>(&padding), sizeof(padding));

	ofs.write(reinterpret_cast<const char*>(data.dense_elements.data()), data.dense_elements.size() * sizeof(float));
	chunk_size = data.dense_elements.size() * sizeof(float);
	padding = calculate_padding(chunk_size);
	printf("Dense Matrix: Size equal to %lu, I need to pad %lu bytes\n", chunk_size, padding);
	ofs.write(reinterpret_cast<const char*>(&padding), sizeof(padding));

	ofs.close();
}

static void batch_convert(const std::filesystem::recursive_directory_iterator& data_dir)
{
	for (const auto& filepath : data_dir) {
		if (filepath.is_regular_file() && requires_conversion(filepath.path())) {
			COOMatrix          mtx = read_mtx(filepath);
			std::vector<float> sparse_vec = coo_to_row_major(mtx);
			std::vector<float> dense_vec = generate_dense(mtx.rows * mtx.cols);

			const std::filesystem::path output_filepath = replace_extension(filepath, ".spmm");
			serialize(output_filepath, {
										   std::move(sparse_vec),
										   std::move(dense_vec),
										   mtx.rows,
										   mtx.cols,
									   });
		}
	}
}

/*
   * Expects a .mtx and looks for a .spmm with the same name
   * converts from coo to row major
   * reads .spmmm
   * compares
 */
static bool unit_test_serialization(const std::filesystem::path& filepath)
{
	if (filepath.extension() != ".mtx") {
		THROW_RUNTIME_ERROR("Unexpected filepath extension, expected: '.mtx'");
	}

	const auto binary_filepath = replace_extension(filepath, ".spmm");

	if (!std::filesystem::exists(binary_filepath)) {
		THROW_RUNTIME_ERROR("Corresponding .spmm not found");
	}

	COOMatrix          mtx = read_mtx(filepath);
	std::vector<float> expected = coo_to_row_major(std::move(mtx));

	SerializationInput actual;

	std::ifstream ifs(binary_filepath, std::ios::binary);

	ifs.read(reinterpret_cast<char*>(&actual.rows), sizeof(actual.rows));
	printf("Read %lu bytes, rows are equal to %d\n", sizeof(actual.rows), actual.rows);
	ifs.read(reinterpret_cast<char*>(&actual.cols), sizeof(actual.cols));
	printf("Read %lu bytes, cols are equal to %d\n", sizeof(actual.cols), actual.cols);
	printf("Reserving %lu bytes of memory for the vector of sparse elements (%u)\n", actual.rows * actual.cols * sizeof(float), actual.rows * actual.cols);
	actual.sparse_elements.resize(actual.rows * actual.cols);
	printf("Reserving %lu bytes of memory for the vector of dense elements (%u)\n", actual.rows * actual.cols * sizeof(float), actual.rows * actual.cols);
	actual.dense_elements.resize(actual.rows * actual.cols);

	printf("Reading %lu bytes, or %u elements of the vector of sparse elements\n", actual.rows * actual.cols * sizeof(float), actual.rows * actual.cols);
	ifs.read(reinterpret_cast<char*>(actual.sparse_elements.data()), actual.rows * actual.cols * sizeof(float));

	printf("Reading %lu bytes, or %u elements of the vector of dense elements\n", actual.rows * actual.cols * sizeof(float), actual.rows * actual.cols);
	ifs.read(reinterpret_cast<char*>(actual.dense_elements.data()), actual.rows * actual.cols * sizeof(float));

	std::cout << expected[0] << " " << actual.sparse_elements[0] << "\n";

	return expected == actual.sparse_elements;
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
	const auto data_path = std::filesystem::current_path() / DATA_DIRECTORY;
	const auto data_dir = std::filesystem::recursive_directory_iterator(std::move(data_path));
	try {
		batch_convert(data_dir);
		assert(unit_test_serialization(data_path / "d50_s2048/d50_s2048.mtx"));
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}
}
