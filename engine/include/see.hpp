#pragma once

#include "board.hpp"
#include "types.hpp"

namespace cortex {

/// Stand-alone SEE for a capture (or EP); non-captures return 0.
int static_exchange_eval(const Board& b, Move m);

}  // namespace cortex
