#pragma once

#include <stdint.h>

// public header

typedef enum
{
	SPMM_STATUS_SUCCESS = 0,
	SPMM_STATUS_NOT_INITIALIZED = 1,
	SPMM_STATUS_ALLOC_FAILED = 2,
	SPMM_STATUS_INVALID_VALUE = 3,
	SPMM_STATUS_ARCH_MISMATCH = 4,
	SPMM_STATUS_MAPPING_ERROR = 5,
	SPMM_STATUS_EXECUTION_FAILED = 6,
	SPMM_STATUS_INTERNAL_ERROR = 7,
	SPMM_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
	SPMM_STATUS_ZERO_PIVOT = 9,
	SPMM_STATUS_NOT_SUPPORTED = 10,
	SPMM_STATUS_INSUFFICIENT_RESOURCES = 11
} SpmmStatus_t;

typedef enum
{
	INDEX_TYPE_16U = 1,
	INDEX_TYPE_32U = 2,
	INDEX_TYPE_64U = 3,
} indexType_t;

typedef enum
{
	DATA_TYPE_F32 = 1,
} dataType_t;

struct Context;
typedef struct Context* Handle_t;

SpmmStatus_t create_handle(Handle_t* handle);
SpmmStatus_t destroy_handle(Handle_t* handle);

struct SpMatDescr;
typedef struct SpMatDescr* SpMatDescr_t;

SpmmStatus_t create_sp_mat_csr(SpMatDescr_t* sp_mat_descr,
	uint32_t                                 rows,
	uint32_t                                 cols,
	uint32_t                                 nnz,
	void*                                    row_ptr,
	void*                                    col_idx,
	void*                                    values,
	indexType_t                              index_type,
	dataType_t                               val_type);

SpmmStatus_t create_sp_mat_csc(SpMatDescr_t* sp_mat_descr,
	uint32_t                                 rows,
	uint32_t                                 cols,
	uint32_t                                 nnz,
	void*                                    col_ptr,
	void*                                    row_idx,
	void*                                    values,
	indexType_t                              index_type,
	dataType_t                               val_type);
