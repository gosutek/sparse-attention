#include <filesystem>

#include "spmm.h"
#include "unit_tests.h"
#include "utils.h"

SpmmTestStatus_t ut_run_tests()
{
	if (!std::filesystem::exists(TEST_CASE_DIR)) {
		fprintf(stderr, "TEST_CASE_DIR=%s doesn't exist", TEST_CASE_DIR);
		return SPMM_TEST_STATUS_INVALID_FILEPATH;
	}

	ExecutionContext_t ctx = NULL;
	exec_ctx_create(&ctx);

	const std::filesystem::path path_base = { TEST_CASE_DIR };

	const std::filesystem::directory_iterator path_csr_to_csc(path_base / "csr_to_csc");

	for (const auto& test_file : path_csr_to_csc) {
		if (test_file.path().extension() == ".ans") {
			continue;
		}

		const std::filesystem::path ans_file(test_file.path().stem() / ".ans");
		if (!std::filesystem::exists(ans_file)) {
			return SPMM_TEST_STATUS_ANSWER_FILE_NOT_FOUND;
		}

		CSR csr = parse_csr_test_case(test_file);
		CSC csc = parse_csc_test_case(ans_file);
	}

	exec_ctx_destroy(ctx);

	return SPMM_TEST_STATUS_OK;
}
