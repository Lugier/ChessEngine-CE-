#include "search.hpp"
#include "eval_classic.hpp"
#include "nnue.hpp"
#include "zobrist.hpp"
#include <algorithm>
#include <chrono>
#include <vector>

namespace cortex {

static constexpr int MATE = 30000;
static std::atomic<bool> g_stop{false};

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
};

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
  ctx.nodes++;
  if (ctx.lim.nodes_max > 0 && ctx.nodes >= ctx.lim.nodes_max) {
    search_set_stop(true);
    return 0;
  }

  Board& b = *ctx.board;
  if (is_rule_draw(b, ctx.rep_stack)) return 0;

  int stand = evaluate_board(b);
  if (stand >= beta) return stand;
  if (stand > alpha) alpha = stand;

  Movelist ml;
  noisy_moves(b, ml);
  int scores[256];
  for (int i = 0; i < ml.count; ++i) scores[i] = mvv_lva(b, ml.moves[i]);
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
        if (te->bound == BOUND_EXACT ||
            (te->bound == BOUND_LOWER && te->score >= beta) ||
            (te->bound == BOUND_UPPER && te->score <= alpha))
          return te->score;
      }
    }
  }

  if (allow_null && depth >= 3 && !in_check && beta == alpha + 1) {
    Square old_ep = b.ep_square;
    b.side = ~b.side;
    b.hash ^= ZOBRIST.side;
    if (old_ep != SQ_NONE) {
      b.hash ^= ZOBRIST.ep[file_of(old_ep)];
      b.ep_square = SQ_NONE;
    }
    int R = 2 + depth / 6;
    int sc = -negamax(ctx, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
    b.side = ~b.side;
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
  for (int i = 0; i < ml.count; ++i) {
    Move m = ml.moves[i];
    if (m == tt_move)
      order_sc[i] = 1'000'000;
    else
      order_sc[i] = mvv_lva(b, m);
  }
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
    if (alpha >= beta) break;
  }

  Bound bd = BOUND_EXACT;
  if (best_sc <= orig_alpha) bd = BOUND_UPPER;
  else if (best_sc >= beta)
    bd = BOUND_LOWER;
  if (ctx.tt && best_mv) ctx.tt->store(key, depth, best_sc, bd, best_mv);
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
  for (int d = 1; d <= lim.depth; ++d) {
    int alpha = -MATE - 1;
    int beta = MATE + 1;
    int best_sc = -MATE - 1;
    Move best_m = best;
    for (int i = 0; i < root.count; ++i) {
      if (search_stopped() || time_up(ctx)) goto done;
      Move m = root.moves[i];
      UndoInfo u;
      b.do_move(m, u);
      ctx.rep_stack.push_back(b.hash);
      int sc = -negamax(ctx, d - 1, -beta, -alpha, 1, true);
      ctx.rep_stack.pop_back();
      b.undo_move(m, u);
      if (search_stopped()) goto done;
      if (sc > best_sc) {
        best_sc = sc;
        best_m = m;
      }
      if (sc > alpha) alpha = sc;
    }
    best = best_m;
    if (search_stopped() || time_up(ctx)) break;
  }
done:
  return best;
}

}  // namespace cortex
