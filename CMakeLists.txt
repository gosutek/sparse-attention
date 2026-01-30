cmake_minimum_required(VERSION 3.18)
project(cute LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpointer-arith -Weffc++ -Wextra -Wconversion -Wsign-conversion -pedantic")

add_executable(cute src/main.cpp)

