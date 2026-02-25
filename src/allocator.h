#if !defined(ALLOCATOR_H)
#define ALLOCATOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "allocator.cuh"
#include "helpers.h"
#include "spmm.h"

typedef struct MemArena
{
	uint64_t reserve_size;
	uint64_t commit_size;

	uint64_t commit_pos;
	uint64_t pos;
} MemArena;

typedef struct ExecCtx
{
	MemArena  host_arena;
	DevArena* dev_arena;
} ExecCtx;

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

SpmmInternalStatus_t mem_arena_host_create(MemArena** const arena, const uint64_t reserve_size, const uint64_t commit_size);
SpmmInternalStatus_t mem_arena_host_destroy(MemArena* arena);

SpmmInternalStatus_t mem_arena_host_push(MemArena* const arena, const uint64_t req_size, void** ptr_out);
void                 mem_arena_host_pop(MemArena* const arena, uint64_t size);
void                 mem_arena_host_pop_at(MemArena* const arena, uint64_t pos);

uint64_t mem_arena_host_pos_get(const MemArena* const arena);

#endif  // ALLOCATOR_H
