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
	uint64_t reserve_size;
	uint64_t commit_size;

	uint64_t commit_pos;
	uint64_t pos;
} MemArena;

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

inline static int32_t mem_arena_create(MemArena** const arena, uint64_t reserve_size);
inline static int32_t mem_arena_destroy(MemArena* arena);

inline static int32_t mem_arena_push(MemArena* const arena, uint64_t size, const void** ptr_out);
inline static int32_t mem_arena_pop(MemArena* const arena, uint64_t size);

inline uint64_t mem_arena_pos_get(const MemArena* const arena);

/*
      * +------------------------------------------------------------------------------+
      * |                             PLATFORM SPECIFIC                                |
      * +------------------------------------------------------------------------------+
*/

inline static uint32_t vm_get_page_size();
inline static void*    vm_reserve(const uint64_t size);
inline static int32_t  vm_release(void* ptr, const uint64_t size);
inline static int32_t  vm_commit(void* addr, const uint64_t size);
inline static void     vm_uncommit();

#endif  // ALLOCATOR_H
