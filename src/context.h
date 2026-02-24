#if !defined(CONTEXT_H)
#define CONTEXT_H

#include "allocator.h"
#include "memory.cuh"
#include "spmm.h"

typedef struct ExecCtx
{
	MemArena  host_arena;
	DevArena* dev_arena;
} ExecCtx;

/*
      * +------------------------------------------------------------------------------+
      * |                                PUBLIC API                                    |
      * +------------------------------------------------------------------------------+
*/

SpmmStatus_t exec_ctx_create(ExecutionContext_t* ctx);
SpmmStatus_t exec_ctx_destroy(ExecutionContext_t ctx);

#endif  // CONTEXT_H
