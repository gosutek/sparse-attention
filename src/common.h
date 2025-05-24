#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

constexpr char    DATA_DIRECTORY[] = "data/";
constexpr uint8_t ALIGNMENT = 128;
constexpr uint8_t ROW_PANEL_SIZE = 32;  // I think this should be the same as TM

constexpr uint8_t TM = 32;
constexpr uint8_t TK = 16;
constexpr uint8_t brick_m = 16;
constexpr uint8_t brick_k = 4;

#define THROW_RUNTIME_ERROR(message) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + message)

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
