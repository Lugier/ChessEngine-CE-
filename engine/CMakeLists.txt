cmake_minimum_required(VERSION 3.16)
project(cortex_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cortex
  src/main.cpp
  src/bitboard.cpp
  src/zobrist.cpp
  src/board.cpp
  src/movegen.cpp
  src/eval_classic.cpp
  src/tt.cpp
  src/search.cpp
  src/nnue.cpp
  src/uci.cpp
)

target_include_directories(cortex PRIVATE include)
target_compile_options(cortex PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
)

# Gemini.md §6.1: Apple Silicon → -mcpu=apple-m2; sonst -march=native
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  target_compile_options(cortex PRIVATE -mcpu=apple-m2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  target_compile_options(cortex PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(cortex PRIVATE -march=native)
endif()
