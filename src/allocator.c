#include <stdio.h>  // TODO: remove
#include <sys/mman.h>
#include <unistd.h>

#include "allocator.h"
#include "helpers.h"
#include "spmm.h"

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

// TODO: Implement an internal error enum and change the return types of these functions
//
// INFO: COMMIT SIZE SHOULD BE PAGE-SIZE ALIGNED AND DERIVED FROM AN ALLOCATION STRATEGY SIMILAR TO VECTOR OR SOMETHING :)

inline static int32_t mem_arena_create(MemArena** const arena, const uint64_t reserve_size, const uint64_t commit_size)
{
	const uint32_t page_size = vm_get_page_size();
	const uint64_t pa_reserve_size = reserve_size + PADDING_POW2(reserve_size, page_size);
	const uint64_t pa_commit_size = commit_size + PADDING_POW2(commit_size, page_size);

	*arena = (MemArena*)vm_reserve(reserve_size);
	if (!(*arena)) {
		return -1;
	}

	int32_t ret_code = vm_commit(*arena, pa_commit_size); /* Allocate for the MemArena members */
	if (ret_code != 0) {
		return -1;
	}

	(*arena)->reserve_size = pa_reserve_size;
	(*arena)->commit_size = pa_commit_size;

	(*arena)->commit_pos = pa_commit_size;
	(*arena)->pos = sizeof **arena;

	return 0;
}

inline static int32_t mem_arena_destroy(MemArena* arena)
{
	return vm_release(arena, arena->reserve_size) == 0;
}

inline static int32_t mem_arena_push(MemArena* const arena, uint64_t size, const void** ptr_out)
{
	// TODO: Fill this
	return 0;
}

inline static int32_t mem_arena_pop(MemArena* const arena, uint64_t size)
{
	// TODO: Fill this
	return 0;
}

// TODO: Do I need this anymore?
inline uint64_t mem_arena_pos_get(const MemArena* const arena)
{
	return arena->pos;
}

#if defined(__linux__)

inline static uint32_t vm_get_page_size()
{
	return (uint32_t)sysconf(_SC_PAGESIZE);
}

// INFO: Mimics malloc in the sense that it returns a NULL ptr on an error instead of an error enum type
inline static void* vm_reserve(const uint64_t size)
{
	void* ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ptr == MAP_FAILED) {
		return NULL;
	}
	return ptr;
}

inline static int32_t vm_release(void* ptr, const uint64_t size)
{
	return munmap(ptr, size) == 0; /* >"It is not an error if the indicated range does not contain any mapped pages" ~ So 'ptr' can be NULL here.*/
}

inline static int32_t vm_commit(void* addr, const uint64_t size)
{
	return mprotect(addr, size, PROT_READ | PROT_WRITE) == 0;
}
inline static int32_t vm_uncommit(void* addr, const uint64_t size)
{
	int32_t ret_code = mprotect(addr, size, PROT_NONE);
	if (ret_code != 0) {
		return -1;
	}
	return madvise(addr, size, MADV_DONTNEED) == 0; /* Subsequent access will result in zero-fill-on-demand pages */
}

#else
#	error "VIRTUAL MEMORY ALLOCATION NOT IMPLEMENTED FOR CURRENT PLATFORM"
#endif

/*
      * +------------------------------------------------------------------------------+
      * |                                PUBLIC API                                    |
      * +------------------------------------------------------------------------------+
*/

SpmmStatus_t
	exec_ctx_create(ExecutionContext_t* ctx)
{
	if (*ctx) {
		return SPMM_STATUS_INVALID_VALUE;
	}
	mem_arena_create(ctx, GIB(1));
	printf("%lu\n", (*ctx)->mem_alloc_pos);
	if (!*ctx) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	return SPMM_STATUS_SUCCESS;
}
SpmmStatus_t exec_ctx_destroy(ExecutionContext_t* ctx)
{
	if (!*ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}
	mem_arena_destroy(*ctx);
	if (*ctx) {
		return SPMM_STATUS_ALLOC_FAILED;
	}
	return SPMM_STATUS_SUCCESS;
}
