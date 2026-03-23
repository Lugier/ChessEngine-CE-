#include "nnue.hpp"
#include <cmath>
#include <fstream>

namespace cortex {

NnueEvaluator g_nnue;

void NnueEvaluator::board_to_features(const Board& b, int8_t* out) {
  for (int i = 0; i < NnueWeights::kInput; ++i) out[i] = 0;
  auto idx = [](Square sq, Piece pc) -> int {
    if (pc == NO_PIECE) return -1;
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    int side_off = c == WHITE ? 0 : 6;
    int p = side_off + int(pt);
    if (p < 0 || p > 11) return -1;
    return int(sq) * 12 + p;
  };
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE) continue;
    int i = idx(sq, pc);
    if (i >= 0) out[i] = 1;
  }
}

int NnueEvaluator::evaluate(const Board& b) const {
  if (!active_ || !W_.ok()) return 0;
  alignas(64) int8_t feat[NnueWeights::kInput];
  board_to_features(b, feat);
  int32_t hidden[NnueWeights::kHidden];
  for (int j = 0; j < NnueWeights::kHidden; ++j) {
    int32_t sum = W_.b1[j];
    const int16_t* row = &W_.w1[j * NnueWeights::kInput];
    for (int i = 0; i < NnueWeights::kInput; ++i) sum += int32_t(row[i]) * feat[i];
    int32_t v = sum > 0 ? sum : 0;
    hidden[j] = v;
  }
  int32_t out = W_.b2;
  for (int j = 0; j < NnueWeights::kHidden; ++j)
    out += int32_t(W_.w2[j]) * hidden[j];
  int sc = int(out / W_.out_scale);
  return b.side == WHITE ? sc : -sc;
}

void NnueEvaluator::load_dummy_zero() {
  W_.w1.assign(size_t(NnueWeights::kHidden * NnueWeights::kInput), 0);
  W_.b1.assign(NnueWeights::kHidden, 0);
  W_.w2.assign(NnueWeights::kHidden, 0);
  W_.b2 = 0;
  W_.out_scale = 16;
  active_ = true;
}

bool NnueEvaluator::load_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[5] = {};
  f.read(magic, 4);
  if (std::string(magic, 4) != "CXN1") return false;
  uint32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  if (ver != 1) return false;
  f.read(reinterpret_cast<char*>(&W_.out_scale), 4);
  W_.w1.resize(NnueWeights::kHidden * NnueWeights::kInput);
  W_.b1.resize(NnueWeights::kHidden);
  W_.w2.resize(NnueWeights::kHidden);
  f.read(reinterpret_cast<char*>(W_.w1.data()),
         W_.w1.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(W_.b1.data()),
         W_.b1.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(W_.w2.data()),
         W_.w2.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(&W_.b2), sizeof(int16_t));
  active_ = W_.ok();
  return active_;
}

}  // namespace cortex
