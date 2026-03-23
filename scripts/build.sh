#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/engine"
: "${CXX:=clang++}"
"$CXX" -std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic \
  -I include -o cortex \
  src/main.cpp src/bitboard.cpp src/zobrist.cpp src/board.cpp \
  src/movegen.cpp src/eval_classic.cpp src/tt.cpp src/search.cpp \
  src/nnue.cpp src/uci.cpp
echo "Built $ROOT/engine/cortex"
