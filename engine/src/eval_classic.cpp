// Hand-crafted eval: piece values + PST; king square is tapered (MG/EG tables)
// using a 0..24 phase from non-pawn material; bishop-pair bonus fades in endgame.
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

static const int PST_KING_EG[64] = {
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0,   0,   -10, -20, -30,
    -30, -10, 20,  30,  30,  20,  -10, -30, -30, -10, 30,  40,  40,  30,  -10, -30,
    -30, -10, 30,  40,  40,  30,  -10, -30, -30, -10, 20,  30,  30,  20,  -10, -30,
    -30, -30, 0,   0,   0,   0,   -30, -30, -50, -30, -30, -30, -30, -30, -50, -50,
};

static int game_phase24(const Board& b) {
  int w = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE) continue;
    PieceType pt = type_of(pc);
    if (pt == PAWN || pt == KING) continue;
    if (pt == KNIGHT || pt == BISHOP)
      w += 1;
    else if (pt == ROOK)
      w += 2;
    else if (pt == QUEEN)
      w += 4;
  }
  return w > 24 ? 24 : w;
}

static int king_pst_tapered(Square sq, Color view, int ph24) {
  int idx = view == WHITE ? int(sq) : int(sq ^ 56);
  int mg = PST_KING_MG[idx];
  int eg = PST_KING_EG[idx];
  int ps = ph24 * 256 / 24;
  return (mg * ps + eg * (256 - ps)) / 256;
}

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
  int ph24 = game_phase24(b);
  int wb = 0, bb = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    if (b.piece[sq] == W_BISHOP) wb++;
    if (b.piece[sq] == B_BISHOP) bb++;
  }
  int ps = ph24 * 256 / 24;
  int pair = (22 * (256 - ps)) / 256;

  int mg = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE) continue;
    PieceType pt = type_of(pc);
    Color col = color_of(pc);
    int pst = (pt == KING) ? king_pst_tapered(sq, col, ph24) : pst_for(pt, sq, col);
    int v = PIECE_VAL[pt] + pst;
    mg += (col == WHITE) ? v : -v;
  }
  if (wb >= 2) mg += pair;
  if (bb >= 2) mg -= pair;

  return b.side == WHITE ? mg : -mg;
}

}  // namespace cortex
