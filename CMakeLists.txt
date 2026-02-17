cmake_minimum_required(VERSION 3.18)
project(cute LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(cute_main
  run/main.cpp
  run/utils.cpp
)

target_link_libraries(cute_main PRIVATE cute)

target_compile_options(cute_main PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
  -Wall
  -Wextra
  -Wpedantic
  >
)

option(MEM_DEBUG "Enable sanitizers" OFF)

if (MEM_DEBUG)
  target_compile_options(cute_main PRIVATE
  -fsanitize=address,undefined
  -fno-omit-frame-pointer
  -g
  )
  target_link_options(cute_main PRIVATE
  -fsanitize=address,undefined
  )
endif()
