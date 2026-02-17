#pragma once

#include <filesystem>

#include "spmm.h"

#define TEST_CASE_DIR "../test_data/unit/"

enum SpmmTestStatus_t
{
	SPMM_TEST_STATUS_TEST_SUCCESS = 0,
	SPMM_TEST_STATUS_TEST_FAILURE = 1,
	SPMM_TEST_STATUS_IO_FAILURE = 2,
};

SpmmTestStatus_t        ut_run_tests();
static SpmmTestStatus_t ut_sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr, SpMatDescr_t sp_csc);
