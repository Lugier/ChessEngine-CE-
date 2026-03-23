// Iterative deepening at root; PVS negamax with TT, null move, LMR, quiescence.
// Draws: 50-move, material, repetition along rep_stack. See docs/ENGINE.md.
#pragma once

#include "board.hpp"
#include "movegen.hpp"
#include "tt.hpp"
#include <atomic>
#include <cstdint>

namespace cortex {

struct SearchLimits {
  int depth = 64;
  int movetime_ms = 0;
  int64_t nodes_max = 0;
};

void search_set_stop(bool s);
bool search_stopped();

Move search_bestmove(Board& b, const SearchLimits& lim, TranspositionTable& tt);

}  // namespace cortex
