#!/usr/bin/env bash
# Gemini.md §6.1: bevorzugt Apple-Silicon-Tuning; Fallback ohne Fehler auf -march=native.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/engine"
: "${CXX:=clang++}"

SOURCES=(
  src/main.cpp src/bitboard.cpp src/zobrist.cpp src/board.cpp
  src/movegen.cpp src/eval_classic.cpp src/tt.cpp src/search.cpp
  src/nnue.cpp src/uci.cpp
)

do_compile() {
  local -a flags=("$@")
  "$CXX" -std=c++17 -O3 "${flags[@]}" -Wall -Wextra -Wpedantic \
    -I include -o cortex "${SOURCES[@]}"
}

if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
  if do_compile -mcpu=apple-m2; then
    echo "Built (flags: -mcpu=apple-m2)"
  elif do_compile -march=native; then
    echo "Built (fallback: -march=native; kein apple-m2-Tuning auf diesem Rechner)"
  else
    exit 1
  fi
else
  do_compile -march=native
  echo "Built (flags: -march=native)"
fi

echo "Binary: $ROOT/engine/cortex"
