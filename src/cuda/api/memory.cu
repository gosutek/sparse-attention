#include "memory.cuh"

SpmmInternalStatus_t dev_mem_arena_create(DevArena** const arena, const uint64_t bsize)
{
	if (*arena) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	if (cudaMalloc(arena, bsize) != cudaSuccess) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	(*arena)->size = bsize;
	(*arena)->pos = sizeof **arena;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t dev_mem_arena_destroy(DevArena* arena)
{
	if (!arena) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	if (cudaFree(arena) != cudaSuccess) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t mem_arena_push(DevArena* const arena, const uint64_t bsize, void** ptr_out)
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
void mem_arena_pop(DevArena* const arena, uint64_t bsize)
{
	bsize = MIN(bsize, arena->pos - sizeof *arena);
	arena->pos -= bsize;
}

void mem_arena_pop_at(DevArena* const arena, uint64_t pos)
{
}

uint64_t mem_arena_pos_get(const DevArena* const arena);

// namespace spmm
// {
// 	void* cuda_malloc_host(size_t b_size)
// 	{
// 		void* ptr = nullptr;
// 		CUDA_CHECK(cudaMallocHost(&ptr, b_size));
// 		return ptr;
// 	}
//
// 	void cuda_dealloc_host(void* ptr)
// 	{
// 		CUDA_CHECK(cudaFreeHost(ptr));
// 	}
//
// 	void cuda_dealloc_device(void* ptr)
// 	{
// 		CUDA_CHECK(cudaFree(ptr));
// 	}
//
// }  // namespace spmm
