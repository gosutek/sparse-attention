#if !defined(CUDA_ALLOCATOR_CUH)
#define CUDA_ALLOCATOR_CUH

#include <stdint.h>

#include "../../helpers.h"

typedef struct DevArena
{
	uint64_t size;

	uint64_t pos;
} DevArena;

SpmmInternalStatus_t mem_arena_dev_create(DevArena** const arena, const uint64_t bsize);
SpmmInternalStatus_t mem_arena_dev_destroy(DevArena* arena);

SpmmInternalStatus_t mem_arena_dev_push(DevArena* const arena, const uint64_t bsize, void** ptr_out);
void                 mem_arena_dev_pop(DevArena* const arena, uint64_t bsize);
void                 mem_arena_dev_pop_at(DevArena* const arena, uint64_t pos);

uint64_t mem_arena_dev_pos_get(const DevArena* const arena);

#endif  // CUDA_ALLOCATOR_CUH
