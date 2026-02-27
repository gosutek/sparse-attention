#include "cuda_allocator.cuh"

SpmmInternalStatus_t mem_arena_dev_create(DevArena* arena, const uint64_t bsize)
{
	if (arena) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	if (cudaMalloc(&arena->d_ptr, bsize) != cudaSuccess) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	arena->size = bsize;
	arena->pos = sizeof *arena;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t mem_arena_dev_destroy(DevArena* arena)
{
	if (!arena->d_ptr) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	if (cudaFree(arena->d_ptr) != cudaSuccess) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}
	arena->d_ptr = nullptr;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t mem_arena_dev_push(DevArena* const arena, const uint64_t bsize, void** ptr_out)
{
	if (!arena) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	const uint64_t pos_aligned = arena->pos + PADDING_POW2(arena->pos, sizeof(void*));
	const uint64_t new_pos = pos_aligned + bsize;

	if (new_pos > arena->size) {
		abort();
	}

	*ptr_out = (uint8_t*)arena + pos_aligned;
	arena->pos = new_pos;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

// WARN: What if bsize isn't aligned?
void mem_arena_dev_pop(DevArena* const arena, uint64_t bsize)
{
	bsize = MIN(bsize, arena->pos - sizeof *arena);
	arena->pos -= bsize;
}

void mem_arena_dev_pop_at(DevArena* const arena, uint64_t pos)
{
	uint64_t size = pos < arena->pos ? arena->pos - pos : 0;
	mem_arena_dev_pop(arena, size);
}

uint64_t mem_arena_dev_pos_get(const DevArena* const arena)
{
	return arena->pos;
}
