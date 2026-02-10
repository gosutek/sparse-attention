#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "helpers.h"
#include "spmm.h"

typedef struct MemArena
{
	uint64_t mem_alloc_pos;
} MemArena;

inline static void mem_arena_create(MemArena** const arena, uint64_t size);
inline static void mem_arena_destroy(MemArena* arena);

inline static void mem_arena_push(MemArena* const arena, uint64_t size, const void** ptr_out);
inline static void mem_arena_push_zero(MemArena* const arena, uint64_t size, void** ptr_out);

inline static void mem_arena_pop(MemArena* const arena, uint64_t size);

inline static uint64_t mem_arena_pos_get(const MemArena* const);

#if defined(__linux__)

inline static uint32_t vm_get_page_size();
inline static void     vm_reserve();
inline static void     vm_release();
inline static void     vm_commit();
inline static void     vm_uncommit();

#endif  // __linux__

#endif  // ALLOCATOR_H
