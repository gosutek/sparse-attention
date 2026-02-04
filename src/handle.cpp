#include "handle.h"
#include "header.h"

SpmmStatus_t create_handle(Handle_t* handle_t)
{
	if (*handle_t != NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}
	*handle_t = static_cast<Handle_t>(malloc(sizeof(Context)));
	if (*handle_t == NULL) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	(*handle_t)->dev_arena = NULL;

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t destroy_handle(Handle_t* handle_t)
{
	if (*handle_t == NULL) {
		return SPMM_STATUS_INVALID_VALUE;
	}
	free(*handle_t);
	if (*handle_t != NULL) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	return SPMM_STATUS_SUCCESS;
}
