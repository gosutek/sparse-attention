#include "allocator.h"
#include "spmm.h"

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

inline void mem_arena_create(MemArena** const arena, uint64_t size)
{
	// TODO: Call virtual memory allocator instead of malloc
	*arena = (MemArena*)malloc(size);
	(*arena)->mem_alloc_pos = sizeof **arena;
}
inline void mem_arena_destroy(MemArena* arena)
{
	// TODO: Swap this to virtual memory allocation
	free(arena);
}

inline void mem_arena_push(MemArena* const arena, uint64_t size, const void** ptr_out)
{
	*ptr_out = arena + arena->mem_alloc_pos;
	arena->mem_alloc_pos += size;
}
inline void mem_arena_push_zero(MemArena* const arena, uint64_t size, void** ptr_out)
{
	*ptr_out = arena + arena->mem_alloc_pos;
	arena->mem_alloc_pos += size;

	memset(*ptr_out, 0, size);
}

inline void mem_arena_pop(MemArena* const arena, uint64_t size)
{
	arena->mem_alloc_pos -= size;
}

inline uint64_t mem_arena_pos_get(const MemArena* const arena)
{
	return arena->mem_alloc_pos;
}

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
	mem_arena_create(ctx, GIB(1));
	if (!*ctx) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	return SPMM_STATUS_SUCCESS;
}
SpmmStatus_t exec_ctx_destroy(ExecutionContext_t* ctx)
{
	if (!*ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}
	mem_arena_destroy(*ctx);
	if (*ctx) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	return SPMM_STATUS_SUCCESS;
}
