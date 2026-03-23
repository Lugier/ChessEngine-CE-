#pragma once

#include "types.hpp"

namespace cortex {

struct Zobrist {
  uint64_t psq[PIECE_NB][SQUARE_NB];
  uint64_t side;
  uint64_t castling[16];
  uint64_t ep[8];
};

extern Zobrist ZOBRIST;

void init_zobrist();

}  // namespace cortex
