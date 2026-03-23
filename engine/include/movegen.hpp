#pragma once

#include "board.hpp"
#include "types.hpp"

namespace cortex {

struct Movelist {
  Move moves[MAX_MOVES];
  int count = 0;
  void add(Move m) { moves[count++] = m; }
};

void generate_legal(const Board& b, Movelist& out);

}  // namespace cortex
