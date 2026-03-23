#include "tt.hpp"

namespace cortex {

TranspositionTable::TranspositionTable(size_t mb) {
  size_t entries = (mb * 1024 * 1024) / sizeof(TTEntry);
  size_t pow2 = 1;
  while (pow2 * 2 <= entries) pow2 *= 2;
  table_.resize(pow2);
  mask_ = pow2 - 1;
}

void TranspositionTable::clear() {
  for (auto& e : table_) e = TTEntry{};
}

TTEntry* TranspositionTable::probe(uint64_t key) {
  TTEntry& e = table_[key & mask_];
  if (e.key == key) return &e;
  return nullptr;
}

void TranspositionTable::store(uint64_t key, int depth, int score, Bound b,
                               Move best) {
  TTEntry& e = table_[key & mask_];
  if (e.key == 0 || e.depth <= depth) {
    e.key = key;
    e.depth = depth;
    e.score = score;
    e.bound = b;
    e.best = best;
  }
}

}  // namespace cortex
