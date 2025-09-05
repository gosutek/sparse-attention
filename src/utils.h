#pragma once

#include <iostream>
#include <vector>

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
