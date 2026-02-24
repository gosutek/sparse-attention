#include "context.h"
#include "allocator.h"
#include "helpers.h"
#include "spmm.h"

/*
      * +------------------------------------------------------------------------------+
      * |                                PUBLIC API                                    |
      * +------------------------------------------------------------------------------+
*/

SpmmStatus_t exec_ctx_create(ExecutionContext_t* ctx)
{
	if (*ctx) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (host_mem_arena_create((MemArena**)(ctx), GIB(1), MIB(1)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	if (host_mem_arena_push((MemArena*)(*ctx), sizeof(void*), (void**)&(*ctx)->dev_arena) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t exec_ctx_destroy(ExecutionContext_t ctx)
{
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (host_mem_arena_destroy((MemArena*)(ctx)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	return SPMM_STATUS_SUCCESS;
}
