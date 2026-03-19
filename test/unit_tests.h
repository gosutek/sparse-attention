#pragma once

#include <filesystem>

#include "spmm.h"

#define TEST_CASE_DIR "../test_data/unit/"

enum SpmmTestStatus_t
{
	SPMM_TEST_STATUS_OK = 0,
	SPMM_TEST_STATUS_INVALID_FILEPATH,
	SPMM_TEST_STATUS_ANSWER_FILE_NOT_FOUND,
	SPMM_TEST_STATUS_INVALID_MATRICES,
	SPMM_TEST_STATUS_INVALID_CONTEXT
};

SpmmTestStatus_t ut_run_tests();
