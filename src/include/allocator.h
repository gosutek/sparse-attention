#pragma once

#include <cassert>

#include "cusparse.h"
#include "header.h"
#include "matrix.h"
#include "memory.cuh"

struct Arena
{
	void*  _base;
	void*  _curr;
	size_t _size;
	size_t _used;
};

SpmmStatus_t arena_init(Arena* arena, size_t size);
SpmmStatus_t arena_free(Arena* arena);
