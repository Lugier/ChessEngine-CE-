// CXN1 loader + 768-sparse full refresh; int16 matmul + ClippedReLU. Not SF-HalfKP.
#include "nnue.hpp"
#include <cmath>
#include <fstream>

namespace cortex {

NnueEvaluator g_nnue;

int NnueEvaluator::piece_feature_index(Square sq, Piece pc, const Board& b) const {
  if (pc == NO_PIECE) return -1;
  if (W_.version == NnueWeights::kVersionLegacy) {
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    int side_off = c == WHITE ? 0 : 6;
    int p = side_off + int(pt);
    if (p < 0 || p > 11) return -1;
    return int(sq) * 12 + p;
  }
  // "HalfKP-like": bucket by side-to-move king square (16 buckets), piece-square planes.
  PieceType pt = type_of(pc);
  if (pt == KING) return -1;
  Square ksq = b.king_sq(b.side);
  if (!is_ok(ksq)) return -1;
  int bucket = int(file_of(ksq) / 2) + 4 * int(rank_of(ksq) / 2);
  int type_no_king = int(pt);  // pawn..queen => 0..4
  if (type_no_king < 0 || type_no_king >= 5) return -1;
  int plane = (color_of(pc) == WHITE ? 0 : 5) + type_no_king;  // 10 planes with color
  return bucket * (64 * 10) + int(sq) * 10 + plane;
}

void NnueEvaluator::board_to_features(const Board& b, int8_t* out) const {
  for (int i = 0; i < W_.input_dim; ++i) out[i] = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE) continue;
    int i = piece_feature_index(sq, pc, b);
    if (i >= 0 && i < W_.input_dim) out[i] = 1;
  }
}

void NnueEvaluator::recompute_acc(const int8_t* feat, int32_t* acc) const {
  for (int j = 0; j < NnueWeights::kHidden; ++j) {
    int32_t sum = W_.b1[j];
    const int16_t* row = &W_.w1[j * W_.input_dim];
    sum += dot_i16_i8(row, feat, W_.input_dim);
    acc[j] = sum;
  }
}

void NnueEvaluator::activate_hidden(const int32_t* acc, int32_t* hidden) const {
  for (int j = 0; j < NnueWeights::kHidden; ++j) {
    int32_t v = acc[j] > 0 ? acc[j] : 0;
    if (v > 32767) v = 32767;
    hidden[j] = v;
  }
}

bool NnueEvaluator::try_incremental_update(const Board& b, int32_t* acc, int8_t* feat) const {
  if (!cache_valid_ || W_.version != NnueWeights::kVersionLegacy) return false;
  if (cache_side_ != b.side) return false;
  int changed = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece oldp = cache_piece_[sq];
    Piece newp = b.piece[sq];
    if (oldp == newp) continue;
    ++changed;
    if (oldp != NO_PIECE) {
      int oldi = piece_feature_index(sq, oldp, b);
      if (oldi >= 0) {
        feat[oldi] = 0;
        for (int j = 0; j < NnueWeights::kHidden; ++j)
          acc[j] -= W_.w1[j * W_.input_dim + oldi];
      }
    }
    if (newp != NO_PIECE) {
      int newi = piece_feature_index(sq, newp, b);
      if (newi >= 0) {
        feat[newi] = 1;
        for (int j = 0; j < NnueWeights::kHidden; ++j)
          acc[j] += W_.w1[j * W_.input_dim + newi];
      }
    }
  }
  if (changed == 0) return true;
  if (changed > 8) return false;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1))
    cache_piece_[sq] = b.piece[sq];
  return true;
}

int NnueEvaluator::evaluate(const Board& b) const {
  if (!active_ || !W_.ok()) return 0;
  int32_t acc[NnueWeights::kHidden];
  int32_t hidden[NnueWeights::kHidden];
  int8_t feat[NnueWeights::kInputKingBucket];
  if (W_.input_dim > NnueWeights::kInputKingBucket) return 0;
  bool inc_ok = false;
  if (cache_valid_) {
    for (int j = 0; j < NnueWeights::kHidden; ++j) acc[j] = cache_acc_[j];
    for (int i = 0; i < W_.input_dim; ++i) feat[i] = cache_feat_[i];
    inc_ok = try_incremental_update(b, acc, feat);
  }
  if (!inc_ok) {
    board_to_features(b, feat);
    recompute_acc(feat, acc);
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) cache_piece_[sq] = b.piece[sq];
  }
  activate_hidden(acc, hidden);
  cache_valid_ = true;
  cache_side_ = b.side;
  for (int j = 0; j < NnueWeights::kHidden; ++j) cache_acc_[j] = acc[j];
  for (int j = 0; j < NnueWeights::kHidden; ++j) cache_hidden_[j] = hidden[j];
  for (int i = 0; i < W_.input_dim; ++i) cache_feat_[i] = feat[i];

  if (W_.out_scale <= 0) return 0;
  int64_t out = W_.b2;
  for (int j = 0; j < NnueWeights::kHidden; ++j)
    out += int64_t(W_.w2[j]) * int64_t(hidden[j]);
  int64_t sc64 = out / int64_t(W_.out_scale);
  if (sc64 > 32000) sc64 = 32000;
  if (sc64 < -32000) sc64 = -32000;
  int sc = int(sc64);
  return b.side == WHITE ? sc : -sc;
}

void NnueEvaluator::notify_position_change() { cache_valid_ = false; }

void NnueEvaluator::load_dummy_zero() {
  W_.version = NnueWeights::kVersionLegacy;
  W_.input_dim = NnueWeights::kInputLegacy;
  W_.w1.assign(size_t(NnueWeights::kHidden * W_.input_dim), 0);
  W_.b1.assign(NnueWeights::kHidden, 0);
  W_.w2.assign(NnueWeights::kHidden, 0);
  W_.b2 = 0;
  W_.out_scale = 16;
  active_ = true;
  cache_valid_ = false;
}

bool NnueEvaluator::load_file(const std::string& path) {
  active_ = false;
  cache_valid_ = false;
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[5] = {};
  f.read(magic, 4);
  std::string mg(magic, 4);
  if (mg != "CXN1" && mg != "CXN2") return false;
  uint32_t ver = 0;
  f.read(reinterpret_cast<char*>(&ver), 4);
  if (ver != 1 && ver != 2) return false;
  W_.version = ver;
  W_.input_dim = (ver == NnueWeights::kVersionKingBucket) ? NnueWeights::kInputKingBucket
                                                           : NnueWeights::kInputLegacy;
  f.read(reinterpret_cast<char*>(&W_.out_scale), 4);
  if (!f || W_.out_scale <= 0) return false;
  W_.w1.resize(NnueWeights::kHidden * W_.input_dim);
  W_.b1.resize(NnueWeights::kHidden);
  W_.w2.resize(NnueWeights::kHidden);
  f.read(reinterpret_cast<char*>(W_.w1.data()),
         W_.w1.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(W_.b1.data()),
         W_.b1.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(W_.w2.data()),
         W_.w2.size() * sizeof(int16_t));
  f.read(reinterpret_cast<char*>(&W_.b2), sizeof(int16_t));
  if (!f) {
    active_ = false;
    return false;
  }
  active_ = W_.ok();
  cache_valid_ = false;
  return active_;
}

}  // namespace cortex
