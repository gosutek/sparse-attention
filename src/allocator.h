#if !defined(ALLOCATOR_H)
#define ALLOCATOR_H

#include "cu_allocator.cuh"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef struct MemArena
	{
		u64 reserve_size;
		u64 commit_size;

		u64 commit_pos;
		u64 pos;
	} MemArena;

	typedef struct ExecCtx
	{
		MemArena host_arena;
		DevArena dev_arena;
	} ExecCtx;

	/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

	SpmmInternalStatus_t mem_arena_host_create(MemArena** const arena, const u64 reserve_size, const u64 commit_size);
	SpmmInternalStatus_t mem_arena_host_destroy(MemArena* arena);

	SpmmInternalStatus_t mem_arena_host_push(MemArena* const arena, const u64 req_size, void** ptr_out);
	void                 mem_arena_host_pop(MemArena* const arena, u64 size);
	void                 mem_arena_host_pop_at(MemArena* const arena, u64 pos);

	u64 mem_arena_host_pos_get(const MemArena* const arena);

#if defined(__cplusplus)
}
#endif

#endif  // ALLOCATOR_H
