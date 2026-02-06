cmake_minimum_required(VERSION 3.18)
project(cute LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(cute STATIC
  src/matrix.c
  src/allocator.c
)

target_compile_options(cute PRIVATE
  $<$<COMPILE_LANGUAGE:C>:
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wpointer-arith
    -Wstrict-prototypes
    -Wmissing-prototypes
    -Wold-style-definition
    -Wshadow
    -Wcast-align
    -Wcast-qual
    -Wwrite-strings
    -Wformat=2
    -Wundef
    >
)

target_include_directories(cute
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/api
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/kernels
)

