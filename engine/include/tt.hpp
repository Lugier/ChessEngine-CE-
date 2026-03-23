#pragma once

#include "types.hpp"
#include <cstddef>
#include <vector>

namespace cortex {

enum Bound : uint8_t { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };

struct TTEntry {
  uint64_t key = 0;
  int depth = 0;
  int score = 0;
  Bound bound = BOUND_NONE;
  Move best = {};
};

class TranspositionTable {
 public:
  explicit TranspositionTable(size_t mb);

  void clear();
  TTEntry* probe(uint64_t key);
  void store(uint64_t key, int depth, int score, Bound b, Move best);

 private:
  std::vector<TTEntry> table_;
  size_t mask_ = 0;
};

}  // namespace cortex
