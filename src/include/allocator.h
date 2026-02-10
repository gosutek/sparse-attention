#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#include "spmm.h"

typedef struct MemArena
{
	uint64_t mem_alloc_pos;
} MemArena;

void mem_arena_create(MemArena** const arena, uint64_t size);
void mem_arena_destroy(MemArena* arena);

void mem_arena_push(MemArena* const arena, uint64_t size, const void** ptr_out);
void mem_arena_push_zero(MemArena* const arena, uint64_t size, void** ptr_out);

void mem_arena_pop(MemArena* const arena, uint64_t size);

uint64_t mem_arena_pos_get(const MemArena* const);

#endif  // ALLOCATOR_H
