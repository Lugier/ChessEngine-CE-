// Static exchange eval on capture square; used to order quiescence captures.
#include "see.hpp"
#include "movegen.hpp"

namespace cortex {

static int mat_piece(PieceType pt) {
  static const int v[] = {100, 320, 330, 500, 900, 20000};
  if (pt < PAWN || pt > KING) return 0;
  return v[pt];
}

static bool is_cap(const Board& b, Move m) {
  if (m.type() == MT_EN_PASSANT) return true;
  return b.piece[m.to_sq()] != NO_PIECE;
}

static int victim_value(const Board& b, Move m) {
  if (m.type() == MT_EN_PASSANT) return mat_piece(PAWN);
  return mat_piece(type_of(b.piece[m.to_sq()]));
}

static int see_recapture(Board& b, Square to) {
  Movelist ml;
  generate_legal(b, ml);
  Move pick{};
  int cheapest = 1'000'000'000;
  for (int i = 0; i < ml.count; ++i) {
    Move m = ml.moves[i];
    if (m.to_sq() != to || !is_cap(b, m)) continue;
    Piece att = b.piece[m.from_sq()];
    int pr = mat_piece(type_of(att));
    if (pr < cheapest) {
      cheapest = pr;
      pick = m;
    }
  }
  if (!pick) return 0;
  int captured = mat_piece(type_of(b.piece[to]));
  UndoInfo u;
  b.do_move(pick, u);
  int s = captured - see_recapture(b, to);
  b.undo_move(pick, u);
  return s;
}

int static_exchange_eval(const Board& b, Move m) {
  if (!is_cap(b, m)) return 0;
  Board c = b;
  int v = victim_value(c, m);
  UndoInfo u;
  c.do_move(m, u);
  return v - see_recapture(c, m.to_sq());
}

}  // namespace cortex
