#if !defined(MEMORY_CUH)
#define MEMORY_CUH

#include <stdint.h>

#include "../../helpers.h"

typedef struct DevArena
{
	uint64_t size;

	uint64_t pos;
} DevArena;

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

SpmmInternalStatus_t mem_arena_create(DevArena** const arena, const uint64_t bsize);
SpmmInternalStatus_t mem_arena_destroy(DevArena* arena);

SpmmInternalStatus_t mem_arena_push(DevArena* const arena, const uint64_t bsize, void** ptr_out);
void                 mem_arena_pop(DevArena* const arena, uint64_t bsize);
void                 mem_arena_pop_at(DevArena* const arena, uint64_t pos);

uint64_t mem_arena_pos_get(const DevArena* const arena);

#endif  // MEMORY_CUH
