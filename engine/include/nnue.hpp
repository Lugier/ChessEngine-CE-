#pragma once

#include "board.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace cortex {

struct NnueWeights {
  static constexpr int kInput = 768;
  static constexpr int kHidden = 256;

  std::vector<int16_t> w1;
  std::vector<int16_t> b1;
  std::vector<int16_t> w2;
  int16_t b2 = 0;
  int32_t out_scale = 16;

  bool ok() const {
    return w1.size() == size_t(kHidden * kInput) && b1.size() == size_t(kHidden) &&
           w2.size() == size_t(kHidden);
  }
};

class NnueEvaluator {
 public:
  bool load_file(const std::string& path);
  void load_dummy_zero();
  bool active() const { return active_; }
  int evaluate(const Board& b) const;

 private:
  bool active_ = false;
  NnueWeights W_;
  static void board_to_features(const Board& b, int8_t* out);
};

extern NnueEvaluator g_nnue;

}  // namespace cortex
