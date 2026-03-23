#include "eval_classic.hpp"

namespace cortex {

static const int PIECE_VAL[PIECE_TYPE_NB] = {100, 320, 330, 500, 900, 20000};

static const int PST_PAWN[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,  50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20,
    30, 30, 20, 10, 10, 5,  5,  10, 25, 25, 10, 5,  5,  0,  0,  0,  0,  20, 20,
    0,  0,  0,  0,  5,  -5, -10, 0,  0,  -10, -5, 5,  0,  0,  0,  0,  0,  0,  0,
};

static const int PST_KNIGHT[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   0, 0,   0,   -20, -40,
    -30, 0,   10,  15,  15,  10,  0,   -30, -30, 5,   15,  20, 20,  15,  5,   -30,
    -30, 0,   15,  20,  20,  15,  0,   -30, -30, 5,   10,  15, 15,  10,  5,   -30,
    -40, -20, 0,   5,   5,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
};

static const int PST_BISHOP[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0,   0,   0,   0,   0,   0,   -10,
    -10, 0,   5,   5,   5,   5,   0,   -10, -10, 5,   5,   10,  10,  5,   5,   -10,
    -10, 0,   10,  10,  10,  10,  0,   -10, -10, 5,   5,   5,   5,   5,   5,   -10,
    -10, 0,   0,   0,   0,   0,   0,   -10, -20, -10, -10, -10, -10, -10, -10, -20,
};

static const int PST_ROOK[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,   //
    5,  10, 10, 10, 10, 10, 10, 5,   //
    -5, 0,  0,  0,  0,  0,  0,  -5,  //
    -5, 0,  0,  0,  0,  0,  0,  -5,  //
    -5, 0,  0,  0,  0,  0,  0,  -5,  //
    -5, 0,  0,  0,  0,  0,  0,  -5,  //
    -5, 0,  0,  0,  0,  0,  0,  -5,  //
    0,  0,  0,  5,  5,  0,  0,  0,   //
};

static const int PST_QUEEN[64] = {
    -20, -10, -10, -5,  -5,  -10, -10, -20, -10, 0,   0,   0,   0,   0,   0,   -10,
    -10, 0,   5,   5,   5,   5,   0,   -10, -5,  0,   5,   5,   5,   5,   0,   -5,
    0,   0,   5,   5,   5,   5,   0,   -5,  -10, 0,   5,   5,   5,   5,   0,   -10,
    -10, 0,   0,   0,   0,   0,   0,   -10, -20, -10, -10, -5,  -5,  -10, -10, -20,
};

static const int PST_KING_MG[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,  0,   0,   0,   0,   20,  20,  20,  30,  10,  0,   0,   10,  30,  20,
};

static int pst_for(PieceType pt, Square sq, Color view) {
  int idx = view == WHITE ? int(sq) : int(sq ^ 56);
  switch (pt) {
    case PAWN:
      return PST_PAWN[idx];
    case KNIGHT:
      return PST_KNIGHT[idx];
    case BISHOP:
      return PST_BISHOP[idx];
    case ROOK:
      return PST_ROOK[idx];
    case QUEEN:
      return PST_QUEEN[idx];
    case KING:
      return PST_KING_MG[idx];
    default:
      return 0;
  }
}

int evaluate_classic(const Board& b) {
  int mg = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE) continue;
    PieceType pt = type_of(pc);
    int v = PIECE_VAL[pt] + pst_for(pt, sq, color_of(pc));
    mg += (color_of(pc) == WHITE) ? v : -v;
  }
  return b.side == WHITE ? mg : -mg;
}

}  // namespace cortex
