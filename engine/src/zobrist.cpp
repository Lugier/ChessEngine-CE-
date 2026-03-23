#include "zobrist.hpp"
#include <random>

namespace cortex {

Zobrist ZOBRIST;

void init_zobrist() {
  std::mt19937_64 rng(0xC0FFEEULL);
  for (auto& row : ZOBRIST.psq)
    for (uint64_t& x : row) x = rng();
  ZOBRIST.side = rng();
  for (uint64_t& x : ZOBRIST.castling) x = rng();
  for (uint64_t& x : ZOBRIST.ep) x = rng();
}

}  // namespace cortex
