#if !defined(SPMM_H)
#define SPMM_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

	/*
    * +------------------------------------------------------------------------------+
    * |                             RETURN CODE ENUMS                                |
    * +------------------------------------------------------------------------------+
  */

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

	/*
    * +------------------------------------------------------------------------------+
    * |                             MATRIX DESCRIPTORS                               |
    * +------------------------------------------------------------------------------+
  */

	// INFO: Off for now
	// typedef enum
	// {
	// 	INDEX_TYPE_16U = 1,
	// 	INDEX_TYPE_32U = 2,
	// 	INDEX_TYPE_64U = 3,
	// } indexType_t;

	// INFO: Off for now
	// typedef enum
	// {
	// 	DATA_TYPE_F32 = 1,
	// 	DATA_TYPE_F64 = 2,
	// } dataType_t;

	// TODO: Could decouple arena from ctx, and have a separate workspace
	// struct that gets initialized at a different point in time
	struct ExecCtx;
	typedef struct ExecCtx* ExecutionContext_t;

	SpmmStatus_t exec_ctx_create(ExecutionContext_t* handle);
	SpmmStatus_t exec_ctx_destroy(ExecutionContext_t ctx);

	struct SpMatDescr;
	typedef struct SpMatDescr* SpMatDescr_t;

	SpmmStatus_t create_sp_mat_csr(ExecutionContext_t ctx, SpMatDescr_t* sp_mat_descr,
		uint32_t  rows,
		uint32_t  cols,
		uint32_t  nnz,
		uint32_t* row_ptr,
		uint32_t* col_idx,
		float*    val);

	SpmmStatus_t create_sp_mat_csc(ExecutionContext_t ctx, SpMatDescr_t* sp_mat_descr,
		uint32_t  rows,
		uint32_t  cols,
		uint32_t  nnz,
		uint32_t* col_ptr,
		uint32_t* row_idx,
		float*    val);

	struct DnMatDescr;
	typedef struct DnMatDescr* DnMatDescr_t;

	SpmmStatus_t create_dn_mat_row_major(DnMatDescr_t* dn_mat_descr,
		uint32_t                                       rows,
		uint32_t                                       cols,
		float*                                         val);

	SpmmStatus_t create_dn_mat_col_major(DnMatDescr_t* dn_mat_descr,
		uint32_t                                       rows,
		uint32_t                                       cols,
		float*                                         val);

	/*
    * +------------------------------------------------------------------------------+
    * |                             MATRIX UTILITIES                                 |
    * +------------------------------------------------------------------------------+
  */

	SpmmStatus_t sp_csr_to_row_major(SpMatDescr_t sp, DnMatDescr_t dn);
	SpmmStatus_t sp_csc_to_col_major(SpMatDescr_t sp, DnMatDescr_t dn);
	SpmmStatus_t sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr, SpMatDescr_t sp_csc);
	SpmmStatus_t sp_csc_to_csr(ExecutionContext_t ctx, SpMatDescr_t sp_csc, SpMatDescr_t sp_csr);

	/*
    * +------------------------------------------------------------------------------+
    * |                                  CUDA OPS                                    |
    * +------------------------------------------------------------------------------+
  */

	SpmmStatus_t spmm(ExecutionContext_t ctx, SpMatDescr_t sp, DnMatDescr_t dn);

#if defined(__cplusplus)
}
#endif

#endif  // SPMM_H
