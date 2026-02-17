#include "unit_tests.h"

SpmmTestStatus_t ut_run_tests()
{
	if (!std::filesystem::exists(TEST_CASE_DIR)) {
		fprintf(stderr, "TEST_CASE_DIR=%s doesn't exist", TEST_CASE_DIR);
		return SPMM_TEST_STATUS_IO_FAILURE;
	}

	return SPMM_TEST_STATUS_TEST_SUCCESS;
}

static SpmmTestStatus_t ut_sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr, SpMatDescr_t sp_csc)
{
}
