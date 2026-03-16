#if !defined(CUDA_ALLOCATOR_CUH)
#define CUDA_ALLOCATOR_CUH

#include <stdint.h>
#include <stdio.h>

#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef struct DevArena
	{
		uint8_t* _d_ptr;

		u64 size;
		u64 pos;
	} DevArena;

	SpmmInternalStatus_t mem_arena_dev_create(DevArena* const arena, const u64 bsize);
	SpmmInternalStatus_t mem_arena_dev_destroy(DevArena* arena);

	SpmmInternalStatus_t mem_arena_dev_push(DevArena* const arena, const u64 bsize, void** ptr_out);
	void                 mem_arena_dev_pop(DevArena* const arena, u64 bsize);
	void                 mem_arena_dev_pop_at(DevArena* const arena, u64 pos);

	u64 mem_arena_dev_pos_get(const DevArena* const arena);

#if defined(__cplusplus)
}
#endif

#endif  // CUDA_ALLOCATOR_CUH
