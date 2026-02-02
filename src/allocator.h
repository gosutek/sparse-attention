#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include "cuda/api/memory.cuh"
#include "matrix.h"

class HostArena
{
private:
	void*  _base;
	size_t _size;

public:
	explicit HostArena(size_t size) : _size(size), _base(cuda_malloc_host(size))
	{
		assert(_base && "HostArena failed to allocate on host");
	}

	~HostArena()
	{
		cuda_dealloc_host(_base);
	}

	HostArena(const HostArena&) = delete;

	HostArena& operator=(const HostArena&) = delete;

	HostArena(HostArena&& other) noexcept : _size(other._size), _base(other._base)
	{
		other._base = nullptr;
		other._size = 0;
	}

	HostArena& operator=(HostArena&& other) noexcept
	{
		if (this != &other) {
			cuda_dealloc_host(_base);

			_size = other._size;
			_base = other._base;

			other._base = nullptr;
			other._size = 0;
		}
		return *this;
	}
};

class DeviceArena
{
private:
	void*  _base;
	size_t _size;

public:
	explicit DeviceArena(size_t size) : _size(size), _base(cuda_malloc_device(size))
	{
		assert(_base && "HostArena failed to allocate on host");
	}

	~DeviceArena()
	{
		cuda_dealloc_device(_base);
	}

	DeviceArena(const DeviceArena&) = delete;

	DeviceArena& operator=(const HostArena&) = delete;

	DeviceArena(DeviceArena&& other) noexcept : _size(other._size), _base(other._base)
	{
		other._base = nullptr;
		other._size = 0;
	}

	DeviceArena& operator=(DeviceArena&& other) noexcept
	{
		if (this != &other) {
			cuda_dealloc_device(_base);

			_size = other._size;
			_base = other._base;

			other._base = nullptr;
			other._size = 0;
		}
		return *this;
	}
};
