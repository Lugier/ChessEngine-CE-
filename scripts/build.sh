#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/engine"
: "${CXX:=clang++}"
# Gemini.md §6.1: Clang für Apple Silicon gezielt tunen (8 GB UMA, NPS).
CPU_FLAGS=(-march=native)
if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  CPU_FLAGS=(-mcpu=apple-m2)
fi
"$CXX" -std=c++17 -O3 "${CPU_FLAGS[@]}" -Wall -Wextra -Wpedantic \
  -I include -o cortex \
  src/main.cpp src/bitboard.cpp src/zobrist.cpp src/board.cpp \
  src/movegen.cpp src/eval_classic.cpp src/tt.cpp src/search.cpp \
  src/nnue.cpp src/uci.cpp
echo "Built $ROOT/engine/cortex"
