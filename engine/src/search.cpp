// Implementation notes: aspiration widens on fail-low/high at root (depth >= 4).
// NNUE+classic blend in evaluate_board keeps small nets from dominating tactics.
// Full rationale and test coverage: docs/ENGINE.md, scripts/verify.sh.
#include "search.hpp"
#include "eval_classic.hpp"
#include "nnue.hpp"
#include "see.hpp"
#include "zobrist.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace cortex {

static constexpr int MATE = 30000;
static constexpr int MAX_PLY = 120;
static constexpr int MATE_IN_MAX = MATE - MAX_PLY;    // scores >= this: win-mate band
static constexpr int MATED_IN_MAX = -MATE + MAX_PLY;  // scores <= this: loss-mate band
static std::atomic<bool> g_stop{false};

// Mate scores depend on search ply; TT must store/retrieve in a ply-invariant form
// (standard trick; avoids wrong cutoffs when the same node is hit at different depths).
static int to_tt(int v, int ply) {
  if (v >= MATE_IN_MAX) return v + ply;
  if (v <= MATED_IN_MAX) return v - ply;
  return v;
}

static int from_tt(int v, int ply) {
  if (v >= MATE_IN_MAX) return v - ply;
  if (v <= MATED_IN_MAX) return v + ply;
  return v;
}

void search_set_stop(bool s) { g_stop = s; }
bool search_stopped() { return g_stop.load(); }

static int evaluate_board(const Board& b) {
  if (g_nnue.active()) {
    int n = g_nnue.evaluate(b);
    int c = evaluate_classic(b);
    return (n * 3 + c) / 4;
  }
  return evaluate_classic(b);
}

static int mvv_lva(const Board& b, Move m) {
  if (m.type() == MT_EN_PASSANT) return 100 - 1;
  Piece victim = b.piece[m.to_sq()];
  Piece attacker = b.piece[m.from_sq()];
  static const int val[] = {100, 320, 330, 500, 900, 20000};
  int v = (victim != NO_PIECE && type_of(victim) <= KING)
              ? val[type_of(victim)]
              : 0;
  int a = val[type_of(attacker)];
  return v * 1000 - a;
}

static void noisy_moves(const Board& b, Movelist& out) {
  Movelist tmp;
  generate_legal(b, tmp);
  out.count = 0;
  for (int i = 0; i < tmp.count; ++i) {
    Move m = tmp.moves[i];
    if (m.type() == MT_PROMOTION || m.type() == MT_EN_PASSANT) {
      out.add(m);
      continue;
    }
    if (b.piece[m.to_sq()] != NO_PIECE) out.add(m);
  }
}

struct SearchCtx {
  Board* board = nullptr;
  TranspositionTable* tt = nullptr;
  int64_t nodes = 0;
  SearchLimits lim;
  std::chrono::steady_clock::time_point start;
  std::vector<uint64_t> rep_stack;
  Move killers[MAX_PLY][2]{};
  int history[2][SQUARE_NB][SQUARE_NB]{};

  void clear_heuristics() {
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
  }
};

static int move_order_score(SearchCtx& ctx, const Board& b, Move m, Move tt_m,
                            int ply) {
  if (m == tt_m) return 2'000'000;
  bool cap = b.piece[m.to_sq()] != NO_PIECE || m.type() == MT_EN_PASSANT;
  if (m.type() == MT_PROMOTION)
    return 1'950'000 - int(m.promo_kind());
  if (cap) return 1'000'000 + mvv_lva(b, m);
  if (ply >= 0 && ply < MAX_PLY) {
    if (m == ctx.killers[ply][0]) return 900'000;
    if (m == ctx.killers[ply][1]) return 899'000;
  }
  int side = b.side == WHITE ? 0 : 1;
  return std::min(500'000, ctx.history[side][(int)m.from_sq()][(int)m.to_sq()]);
}

static bool is_rule_draw(const Board& b, const std::vector<uint64_t>& rep) {
  if (b.halfmove >= 100) return true;
  if (b.is_material_draw()) return true;
  int c = 0;
  for (uint64_t k : rep)
    if (k == b.hash) c++;
  return c >= 3;
}

static bool time_up(const SearchCtx& c) {
  if (c.lim.movetime_ms <= 0) return false;
  auto now = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - c.start)
                .count();
  return ms >= c.lim.movetime_ms;
}

static int qsearch(SearchCtx& ctx, int alpha, int beta, int ply) {
  if (search_stopped() || time_up(ctx)) return 0;
  if (ply >= MAX_PLY) return evaluate_board(*ctx.board);
  ctx.nodes++;
  if (ctx.lim.nodes_max > 0 && ctx.nodes >= ctx.lim.nodes_max) {
    search_set_stop(true);
    return 0;
  }

  Board& b = *ctx.board;
  if (is_rule_draw(b, ctx.rep_stack)) return 0;
  bool in_check = b.in_check();

  if (in_check) {
    Movelist ml;
    generate_legal(b, ml);
    if (ml.count == 0) return -MATE + ply;
    for (int i = 0; i < ml.count; ++i) {
      Move m = ml.moves[i];
      UndoInfo u;
      b.do_move(m, u);
      ctx.rep_stack.push_back(b.hash);
      int sc = -qsearch(ctx, -beta, -alpha, ply + 1);
      ctx.rep_stack.pop_back();
      b.undo_move(m, u);
      if (search_stopped()) return 0;
      if (sc >= beta) return sc;
      if (sc > alpha) alpha = sc;
    }
    return alpha;
  }

  int stand = evaluate_board(b);
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

  Movelist ml;
  noisy_moves(b, ml);
  int scores[256];
  Move no_tt{};
  for (int i = 0; i < ml.count; ++i) {
    int see = static_exchange_eval(b, ml.moves[i]);
    scores[i] = see * 500'000 + move_order_score(ctx, b, ml.moves[i], no_tt, ply);
  }
  for (int a = 0; a < ml.count; ++a)
    for (int j = a + 1; j < ml.count; ++j)
      if (scores[j] > scores[a]) {
        std::swap(scores[a], scores[j]);
        std::swap(ml.moves[a], ml.moves[j]);
      }

  for (int i = 0; i < ml.count; ++i) {
    Move m = ml.moves[i];
    UndoInfo u;
    b.do_move(m, u);
    ctx.rep_stack.push_back(b.hash);
    int sc = -qsearch(ctx, -beta, -alpha, ply + 1);
    ctx.rep_stack.pop_back();
    b.undo_move(m, u);
    if (search_stopped()) return 0;
    if (sc >= beta) return sc;
    if (sc > alpha) alpha = sc;
  }
  return alpha;
}

static int negamax(SearchCtx& ctx, int depth, int alpha, int beta, int ply,
                   bool allow_null) {
  if (search_stopped() || time_up(ctx)) return 0;
  if (ply >= MAX_PLY) return evaluate_board(*ctx.board);
  ctx.nodes++;
  if (ctx.lim.nodes_max > 0 && ctx.nodes >= ctx.lim.nodes_max) {
    search_set_stop(true);
    return 0;
  }

  Board& b = *ctx.board;
  bool in_check = b.in_check();

  if (is_rule_draw(b, ctx.rep_stack)) return 0;

  if (depth <= 0 && !in_check) return qsearch(ctx, alpha, beta, ply);

  uint64_t key = b.hash;
  Move tt_move{};
  if (ctx.tt) {
    if (TTEntry* te = ctx.tt->probe(key)) {
      if (te->best) tt_move = te->best;
      if (te->depth >= depth) {
        int tsc = from_tt(te->score, ply);
        if (te->bound == BOUND_EXACT ||
            (te->bound == BOUND_LOWER && tsc >= beta) ||
            (te->bound == BOUND_UPPER && tsc <= alpha))
          return tsc;
      }
    }
  }

  bool has_non_pawn_material = false;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE || color_of(pc) != b.side) continue;
    PieceType pt = type_of(pc);
    if (pt != PAWN && pt != KING) {
      has_non_pawn_material = true;
      break;
    }
  }
  if (allow_null && depth >= 3 && !in_check && beta == alpha + 1 &&
      has_non_pawn_material) {
    Square old_ep = b.ep_square;
    int old_half = b.halfmove;
    int old_full = b.fullmove;
    b.side = ~b.side;
    b.halfmove++;
    if (b.side == WHITE) b.fullmove++;
    b.hash ^= ZOBRIST.side;
    if (old_ep != SQ_NONE) {
      b.hash ^= ZOBRIST.ep[file_of(old_ep)];
      b.ep_square = SQ_NONE;
    }
    ctx.rep_stack.push_back(b.hash);
    int R = 2 + depth / 6;
    int sc = -negamax(ctx, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
    ctx.rep_stack.pop_back();
    b.side = ~b.side;
    b.halfmove = old_half;
    b.fullmove = old_full;
    b.hash ^= ZOBRIST.side;
    b.ep_square = old_ep;
    if (old_ep != SQ_NONE) b.hash ^= ZOBRIST.ep[file_of(old_ep)];
    if (sc >= beta) return beta;
  }

  Movelist ml;
  generate_legal(b, ml);
  if (ml.count == 0) {
    if (in_check) return -MATE + ply;
    return 0;
  }

  int best_sc = -MATE - 1;
  Move best_mv{};
  int orig_alpha = alpha;

  int order_sc[256];
  for (int i = 0; i < ml.count; ++i)
    order_sc[i] = move_order_score(ctx, b, ml.moves[i], tt_move, ply);
  for (int a = 0; a < ml.count; ++a)
    for (int j = a + 1; j < ml.count; ++j)
      if (order_sc[j] > order_sc[a]) {
        std::swap(order_sc[a], order_sc[j]);
        std::swap(ml.moves[a], ml.moves[j]);
      }

  int moves_done = 0;
  for (int i = 0; i < ml.count; ++i) {
    Move m = ml.moves[i];
    bool capture = b.piece[m.to_sq()] != NO_PIECE || m.type() == MT_EN_PASSANT;
    UndoInfo u;
    b.do_move(m, u);
    ctx.rep_stack.push_back(b.hash);
    int ext = in_check ? 1 : 0;
    int new_depth = depth - 1 + ext;
    int sc;
    if (moves_done == 0)
      sc = -negamax(ctx, new_depth, -beta, -alpha, ply + 1, true);
    else {
      int R = 0;
      if (depth >= 3 && moves_done >= 4 && !in_check && !capture &&
          m.type() != MT_PROMOTION)
        R = 1 + (moves_done > 8 ? 1 : 0);
      sc = -negamax(ctx, new_depth - R, -alpha - 1, -alpha, ply + 1, true);
      if (sc > alpha && sc < beta)
        sc = -negamax(ctx, new_depth, -beta, -alpha, ply + 1, true);
    }
    ctx.rep_stack.pop_back();
    b.undo_move(m, u);
    moves_done++;
    if (search_stopped()) return 0;
    if (sc > best_sc) {
      best_sc = sc;
      best_mv = m;
    }
    if (sc > alpha) alpha = sc;
    if (alpha >= beta) {
      if (!capture && m.type() != MT_PROMOTION && ply < MAX_PLY) {
        if (m != ctx.killers[ply][0]) {
          ctx.killers[ply][1] = ctx.killers[ply][0];
          ctx.killers[ply][0] = m;
        }
        int bonus = depth * depth;
        int mf = (int)m.from_sq(), mt = (int)m.to_sq();
        Color mover = b.side;
        int& h = ctx.history[mover == WHITE ? 0 : 1][mf][mt];
        h = std::min(8192, h + bonus);
      }
      break;
    }
  }

  Bound bd = BOUND_EXACT;
  if (best_sc <= orig_alpha) bd = BOUND_UPPER;
  else if (best_sc >= beta)
    bd = BOUND_LOWER;
  if (ctx.tt && best_mv)
    ctx.tt->store(key, depth, to_tt(best_sc, ply), bd, best_mv);
  return best_sc;
}

Move search_bestmove(Board& b, const SearchLimits& lim, TranspositionTable& tt) {
  search_set_stop(false);
  SearchCtx ctx;
  ctx.board = &b;
  ctx.tt = &tt;
  ctx.lim = lim;
  ctx.start = std::chrono::steady_clock::now();
  ctx.rep_stack = b.game_rep_keys;

  Movelist root;
  generate_legal(b, root);
  if (root.count == 0) return Move{};

  Move best = root.moves[0];
  int prev_sc = 0;
  for (int d = 1; d <= lim.depth; ++d) {
    ctx.clear_heuristics();
    int alpha = -MATE - 1;
    int beta = MATE + 1;
    if (d >= 4) {
      int delta = 32 + std::abs(prev_sc) / 32;
      delta = std::min(delta, MATE / 4);
      alpha = std::max(-MATE + 100, prev_sc - delta);
      beta = std::min(MATE - 100, prev_sc + delta);
    }

    int orig_alpha = alpha;
    int orig_beta = beta;
  redo_root:
    int best_sc = -MATE - 1;
    Move best_m = best;
    for (int i = 0; i < root.count; ++i) {
      if (search_stopped() || time_up(ctx)) goto done;
      Move m = root.moves[i];
      UndoInfo u;
      b.do_move(m, u);
      ctx.rep_stack.push_back(b.hash);
      int sc;
      if (i == 0)
        sc = -negamax(ctx, d - 1, -beta, -alpha, 1, true);
      else {
        sc = -negamax(ctx, d - 1, -alpha - 1, -alpha, 1, true);
        if (sc > alpha && sc < beta)
          sc = -negamax(ctx, d - 1, -beta, -alpha, 1, true);
      }
      ctx.rep_stack.pop_back();
      b.undo_move(m, u);
      if (search_stopped()) goto done;
      if (sc > best_sc) {
        best_sc = sc;
        best_m = m;
      }
      if (sc > alpha) alpha = sc;
    }

    if (d >= 4 && best_sc <= orig_alpha && orig_alpha > -MATE + 200) {
      alpha = -MATE - 1;
      beta = MATE + 1;
      orig_alpha = alpha;
      orig_beta = beta;
      goto redo_root;
    }
    if (d >= 4 && best_sc >= orig_beta && orig_beta < MATE - 200) {
      alpha = -MATE - 1;
      beta = MATE + 1;
      orig_alpha = alpha;
      orig_beta = beta;
      goto redo_root;
    }

    prev_sc = best_sc;
    best = best_m;
    if (search_stopped() || time_up(ctx)) break;
  }
done:
  return best;
}

}  // namespace cortex
