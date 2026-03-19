#if !defined(SPMM_H)
#define SPMM_H

#if defined(__cplusplus)
extern "C"
{
#endif

	/*
    * +------------------------------------------------------------------------------+
    * |                                  ENUMS                                       |
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

	typedef enum
	{
		SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_BLOCK = 0,
		SPMM_KERNEL_TYPE_ELEMWISE_NAIVE_SMEM = 1,
		SPMM_KERNEL_TYPE_NNZWISE_COALESCED = 2,
		SPMM_KERNEL_TYPE_NNZWISE_COALESCED_NO_SMEM = 3,
		SPMM_KERNEL_TYPE_NNZWISE_VECTORIZED = 4,
		SPMM_KERNEL_TYPE_COLUMN_TILING_NNZWISE = 5,
	} SpmmKernelType_t;

	// TODO: Make this into a boolean instead
	typedef enum
	{
		SPMM_KERNEL_INVERT = 0,
		SPMM_KERNEL_NO_INVERT = 1,
	} SpmmInvert_t;

	const char* spmm_get_error_name(SpmmStatus_t err);

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

	SpmmStatus_t sp_csr_create(ExecutionContext_t ctx, SpMatDescr_t* sp,
		uint32_t  rows,
		uint32_t  cols,
		uint32_t  nnz,
		uint32_t* row_ptr,
		uint32_t* col_idx,
		float*    val);

	SpmmStatus_t sp_csr_get(SpMatDescr_t sp,
		uint32_t*                        rows,
		uint32_t*                        cols,
		uint32_t*                        nnz,
		uint32_t**                       row_ptr,
		uint32_t**                       col_idx,
		float**                          val);

	SpmmStatus_t sp_csc_create(ExecutionContext_t ctx, SpMatDescr_t* sp,
		uint32_t  rows,
		uint32_t  cols,
		uint32_t  nnz,
		uint32_t* col_ptr,
		uint32_t* row_idx,
		float*    val);

	SpmmStatus_t sp_csc_get(SpMatDescr_t sp,
		uint32_t*                        rows,
		uint32_t*                        cols,
		uint32_t*                        nnz,
		uint32_t**                       col_ptr,
		uint32_t**                       row_idx,
		float**                          val);

	struct DnMatDescr;
	typedef struct DnMatDescr* DnMatDescr_t;

	SpmmStatus_t dn_rm_create(ExecutionContext_t ctx, DnMatDescr_t* dn,
		uint32_t rows,
		uint32_t cols,
		float*   val);

	SpmmStatus_t dn_rm_get(DnMatDescr_t dn,
		uint32_t*                       rows,
		uint32_t*                       cols,
		float**                         val);

	SpmmStatus_t dn_cm_create(ExecutionContext_t ctx, DnMatDescr_t* dn,
		uint32_t rows,
		uint32_t cols,
		float*   val);

	SpmmStatus_t dn_cm_get(DnMatDescr_t dn,
		uint32_t*                       rows,
		uint32_t*                       cols,
		float**                         val);

	/*
    * +------------------------------------------------------------------------------+
    * |                             MATRIX UTILITIES                                 |
    * +------------------------------------------------------------------------------+
  */

	SpmmStatus_t sp_csr_to_row_major(ExecutionContext_t const ctx, SpMatDescr_t sp, DnMatDescr_t dn);
	SpmmStatus_t sp_csc_to_col_major(SpMatDescr_t sp, DnMatDescr_t dn);
	SpmmStatus_t sp_csr_to_csc(ExecutionContext_t ctx, SpMatDescr_t sp_csr, SpMatDescr_t sp_csc);
	SpmmStatus_t sp_csc_to_csr(ExecutionContext_t ctx, SpMatDescr_t sp_csc, SpMatDescr_t sp_csr);

	/*
    * +------------------------------------------------------------------------------+
    * |                                  CUDA OPS                                    |
    * +------------------------------------------------------------------------------+
  */

	SpmmStatus_t spmm(ExecutionContext_t ctx, SpMatDescr_t sp, DnMatDescr_t dn, DnMatDescr_t res, SpmmKernelType_t kernel_type, SpmmInvert_t invert);
#if defined(__cplusplus)
}
#endif

#endif  // SPMM_H
