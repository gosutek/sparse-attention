#include "cuda_allocator.cuh"

#include <iostream>

#if defined(__cplusplus)
extern "C"
{
#endif

	SpmmInternalStatus_t mem_arena_dev_create(DevArena* arena, const u64 bsize)
	{
		if (arena->_d_ptr) {
			return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
		}

		CHECK_CUDA(cudaMalloc(&arena->_d_ptr, bsize));

		arena->size = bsize;
		arena->pos = sizeof *arena;

		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t mem_arena_dev_destroy(DevArena* arena)
	{
		if (!arena->_d_ptr) {
			return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
		}

		if (cudaFree(arena->_d_ptr) != cudaSuccess) {
			return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
		}
		arena->_d_ptr = nullptr;

		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	SpmmInternalStatus_t mem_arena_dev_push(DevArena* const arena, const u64 bsize, void** ptr_out)
	{
		if (!arena) {
			return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
		}

		const u64 pos_aligned = arena->pos + PADDING_POW2(arena->pos, sizeof(void*));
		const u64 new_pos = pos_aligned + bsize;

		if (new_pos > arena->size) {
			abort();
		}

		*ptr_out = arena->_d_ptr + pos_aligned;
		arena->pos = new_pos;

		return SPMM_INTERNAL_STATUS_SUCCESS;
	}

	// WARN: What if bsize isn't aligned?
	void mem_arena_dev_pop(DevArena* const arena, u64 bsize)
	{
		bsize = MIN(bsize, arena->pos - sizeof *arena);
		arena->pos -= bsize;
	}

	void mem_arena_dev_pop_at(DevArena* const arena, u64 pos)
	{
		u64 size = pos < arena->pos ? arena->pos - pos : 0;
		mem_arena_dev_pop(arena, size);
	}

	u64 mem_arena_dev_pos_get(const DevArena* const arena)
	{
		return arena->pos;
	}

#if defined(__cplusplus)
}
#endif
