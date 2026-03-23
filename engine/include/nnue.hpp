#pragma once

#include "board.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace cortex {

struct NnueWeights {
  static constexpr int kInputLegacy = 768;
  static constexpr int kInputKingBucket = 10240;  // 16 king buckets * 64 * 10
  static constexpr int kHidden = 256;
  static constexpr uint32_t kVersionLegacy = 1;
  static constexpr uint32_t kVersionKingBucket = 2;

  int input_dim = kInputLegacy;
  uint32_t version = kVersionLegacy;

  std::vector<int16_t> w1;
  std::vector<int16_t> b1;
  std::vector<int16_t> w2;
  int16_t b2 = 0;
  int32_t out_scale = 16;

  bool ok() const {
    return input_dim > 0 && w1.size() == size_t(kHidden * input_dim) &&
           b1.size() == size_t(kHidden) &&
           w2.size() == size_t(kHidden);
  }
};

class NnueEvaluator {
 public:
  bool load_file(const std::string& path);
  void load_dummy_zero();
  bool active() const { return active_; }
  int evaluate(const Board& b) const;
  void notify_position_change();

 private:
  bool active_ = false;
  NnueWeights W_;
  mutable bool cache_valid_ = false;
  mutable Piece cache_piece_[SQUARE_NB]{};
  mutable Color cache_side_ = WHITE;
  mutable int32_t cache_hidden_[NnueWeights::kHidden]{};
  mutable int8_t cache_feat_[NnueWeights::kInputKingBucket]{};

  int piece_feature_index(Square sq, Piece pc, const Board& b) const;
  void board_to_features(const Board& b, int8_t* out) const;
  void recompute_hidden(const int8_t* feat, int32_t* hidden) const;
  bool try_incremental_update(const Board& b, int32_t* hidden, int8_t* feat) const;
};

extern NnueEvaluator g_nnue;
int32_t dot_i16_i8(const int16_t* a, const int8_t* b, int n);

}  // namespace cortex
