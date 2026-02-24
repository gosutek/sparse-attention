#include "allocator.h"
#include "helpers.h"

#include "spmm.h"

/*
      * +------------------------------------------------------------------------------+
      * |                             PLATFORM SPECIFIC                                |
      * +------------------------------------------------------------------------------+
*/

#if defined(__linux__)

static uint32_t vm_get_page_size(void)
{
	return (uint32_t)sysconf(_SC_PAGESIZE);
}

// INFO: Mimics malloc in the sense that it returns a NULL ptr on an error instead of an error enum type
static void* vm_reserve(const uint64_t size)
{
	void* ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ptr == MAP_FAILED) {
		return NULL;
	}
	return ptr;
}

static int32_t vm_release(void* ptr, const uint64_t size)
{
	return munmap(ptr, size) == 0; /* >"It is not an error if the indicated range does not contain any mapped pages" ~ So 'ptr' can be NULL here.*/
}

static int32_t vm_commit(void* addr, const uint64_t size)
{
	return mprotect(addr, size, PROT_READ | PROT_WRITE) == 0;
}

static int32_t vm_uncommit(void* addr, const uint64_t size)
{
	int32_t ret_code = mprotect(addr, size, PROT_NONE);
	if (ret_code != 0) {
		return -1;
	}
	return madvise(addr, size, MADV_DONTNEED) == 0; /* Subsequent access will result in zero-fill-on-demand pages */
}

SpmmStatus_t exec_ctx_create(ExecutionContext_t* ctx)
{
	if (*ctx) {
		return SPMM_STATUS_INVALID_VALUE;
	}

	if (host_mem_arena_create((MemArena**)(ctx), GIB(1), MIB(1)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	if (host_mem_arena_push((MemArena*)(*ctx), sizeof(void*), (void**)&(*ctx)->dev_arena) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	return SPMM_STATUS_SUCCESS;
}

SpmmStatus_t exec_ctx_destroy(ExecutionContext_t ctx)
{
	if (!ctx) {
		return SPMM_STATUS_NOT_INITIALIZED;
	}

	if (host_mem_arena_destroy((MemArena*)(ctx)) != SPMM_INTERNAL_STATUS_SUCCESS) {
		return SPMM_STATUS_ALLOC_FAILED;
	}

	return SPMM_STATUS_SUCCESS;
}

#else
#error "VIRTUAL MEMORY ALLOCATION NOT IMPLEMENTED FOR CURRENT PLATFORM"
#endif

/*
      * +------------------------------------------------------------------------------+
      * |                                INTERNALS                                     |
      * +------------------------------------------------------------------------------+
*/

// TODO: Implement an internal error enum and change the return types of these functions
//
// INFO: COMMIT SIZE SHOULD BE PAGE-SIZE ALIGNED AND DERIVED FROM AN ALLOCATION STRATEGY SIMILAR TO VECTOR OR SOMETHING :)

SpmmInternalStatus_t host_mem_arena_create(MemArena** const arena, const uint64_t reserve_size, const uint64_t commit_size)
{
	// TODO: Debug print these at some point to ensure correctness.
	const uint32_t page_size = vm_get_page_size();
	const uint64_t pa_reserve_size = reserve_size + PADDING_POW2(reserve_size, page_size);
	const uint64_t pa_commit_size = commit_size + PADDING_POW2(commit_size, page_size);

	*arena = (MemArena*)vm_reserve(reserve_size);
	if (!(*arena)) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	if (!vm_commit(*arena, pa_commit_size)) { /* Allocate for the MemArena members */
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}

	(*arena)->reserve_size = pa_reserve_size;
	(*arena)->commit_size = pa_commit_size;

	(*arena)->commit_pos = pa_commit_size;
	(*arena)->pos = sizeof **arena;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t host_mem_arena_destroy(MemArena* arena)
{
	if (!vm_release(arena, arena->reserve_size)) {
		return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
	}
	return SPMM_INTERNAL_STATUS_SUCCESS;
}

SpmmInternalStatus_t host_mem_arena_push(MemArena* const arena, const uint64_t req_size, void** ptr_out)
{
	const uint64_t aligned_pos = arena->pos + PADDING_POW2(arena->pos, sizeof(void*)); /* the pointer returned should be naturally aligned */
	const uint64_t new_pos = aligned_pos + req_size;

	if (new_pos > arena->reserve_size) {
		abort();
	} else if (new_pos > arena->commit_pos) {
		const uint64_t commit_size = CEIL_DIVI(new_pos, arena->commit_size);
		if (commit_size > arena->reserve_size) {
			abort();
		}

		if (!vm_commit((uint8_t*)arena + arena->commit_pos, commit_size)) {
			return SPMM_INTERNAL_STATUS_MEMOP_FAIL;
		}
		arena->commit_pos += arena->commit_size;
	}

	*ptr_out = (uint8_t*)arena + aligned_pos;
	arena->pos = new_pos;

	return SPMM_INTERNAL_STATUS_SUCCESS;
}

void host_mem_arena_pop(MemArena* const arena, uint64_t size)
{
	// TODO: Should I null check the ptr here?
	size = MIN(size, arena->pos - sizeof *arena); /* don't dealloc MemArena members */
	arena->pos -= size;
}

void host_mem_arena_pop_at(MemArena* const arena, uint64_t pos)
{
	uint64_t size = pos < arena->pos ? arena->pos - pos : 0;
	host_mem_arena_pop(arena, size);
}

// TODO: Do I need this anymore?
uint64_t host_mem_arena_pos_get(const MemArena* const arena)
{
	return arena->pos;
}
