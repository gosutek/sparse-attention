#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "handle.h"

bool verify_res(const float* const actual, const float* const expected, size_t n);

inline size_t calc_padding_bytes(size_t b_size, size_t alignment_bytes)
{
	return (alignment_bytes - (b_size & (alignment_bytes - 1))) & (alignment_bytes - 1);
}

template <typename T>
std::ostream& operator<<(std::ostream& out_stream, const std::vector<T>& vec)
{
	out_stream << "{";
	for (size_t i = 0; i < vec.size(); ++i) {
		out_stream << vec[i];
		if (i < vec.size() - 1) {
			out_stream << ", ";
		}
	}
	out_stream << "}";
	return out_stream;
}

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& out_stream, const std::array<T, N>& arr)
{
	out_stream << "[";
	for (size_t i = 0; i < arr.size(); ++i) {
		out_stream << arr[i];
		if (i < arr.size() - 1) {
			out_stream << ", ";
		}
	}
	out_stream << "]";
	return out_stream;
}

std::string construct_path(const std::filesystem::path base_path, const BodyType bt, const AttentionMechanism am, const size_t layer);
