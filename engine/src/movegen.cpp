// Piece moves + legality filter (king not left in check).
#include "movegen.hpp"
#include "bitboard.hpp"

namespace cortex {

static void add_pawn_moves(const Board& b, Square from, Movelist& out) {
  Color us = b.side;
  int r = rank_of(from);
  int push = us == WHITE ? 8 : -8;

  Square one = Square(int(from) + push);
  if (is_ok(one) && b.piece[one] == NO_PIECE) {
    int promo_rank = us == WHITE ? 7 : 0;
    if (rank_of(one) == promo_rank) {
      for (PromoKind pk :
           {PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN})
        out.add(Move(from, one, MT_PROMOTION, pk));
    } else
      out.add(Move(from, one, MT_NORMAL));
    Square two = Square(int(from) + 2 * push);
    int start_rank = us == WHITE ? 1 : 6;
    if (r == start_rank && b.piece[two] == NO_PIECE)
      out.add(Move(from, two, MT_NORMAL));
  }

  Bitboard caps = PAWN_ATTACKS[us][from] & b.occupied(~us);
  while (caps) {
    Square to = Square(lsb(caps));
    caps &= caps - 1;
    int pr = us == WHITE ? 7 : 0;
    if (rank_of(to) == pr) {
      for (PromoKind pk :
           {PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN})
        out.add(Move(from, to, MT_PROMOTION, pk));
    } else
      out.add(Move(from, to, MT_NORMAL));
  }

  if (b.ep_square != SQ_NONE &&
      (PAWN_ATTACKS[us][from] & square_bb(b.ep_square)))
    out.add(Move(from, b.ep_square, MT_EN_PASSANT));
}

static void add_piece_moves(const Board& b, Square from, Piece pc, Movelist& out) {
  Bitboard occ = b.occupied();
  PieceType pt = type_of(pc);
  if (pt == PAWN) {
    add_pawn_moves(b, from, out);
    return;
  }
  Bitboard att = attacks_from(pc, from, occ);
  Bitboard ours = b.occupied(b.side);
  att &= ~ours;
  while (att) {
    Square to = Square(lsb(att));
    att &= att - 1;
    out.add(Move(from, to, MT_NORMAL));
  }
}

static void add_castling(const Board& b, Movelist& out) {
  Color us = b.side;
  if (b.in_check()) return;
  Bitboard occ = b.occupied();

  if (us == WHITE) {
    if ((b.castling & Board::CASTLE_WK) && !(occ & ((1ULL << int(SQ_F1)) |
                                                     (1ULL << int(SQ_G1))))) {
      if (!b.is_attacked(SQ_F1, BLACK) && !b.is_attacked(SQ_G1, BLACK))
        out.add(Move(SQ_E1, SQ_G1, MT_CASTLING));
    }
    if ((b.castling & Board::CASTLE_WQ) && !(occ & ((1ULL << int(SQ_D1)) |
                                                     (1ULL << int(SQ_C1)) |
                                                     (1ULL << int(SQ_B1))))) {
      if (!b.is_attacked(SQ_D1, BLACK) && !b.is_attacked(SQ_C1, BLACK))
        out.add(Move(SQ_E1, SQ_C1, MT_CASTLING));
    }
  } else {
    if ((b.castling & Board::CASTLE_BK) && !(occ & ((1ULL << int(SQ_F8)) |
                                                     (1ULL << int(SQ_G8))))) {
      if (!b.is_attacked(SQ_F8, WHITE) && !b.is_attacked(SQ_G8, WHITE))
        out.add(Move(SQ_E8, SQ_G8, MT_CASTLING));
    }
    if ((b.castling & Board::CASTLE_BQ) && !(occ & ((1ULL << int(SQ_D8)) |
                                                     (1ULL << int(SQ_C8)) |
                                                     (1ULL << int(SQ_B8))))) {
      if (!b.is_attacked(SQ_D8, WHITE) && !b.is_attacked(SQ_C8, WHITE))
        out.add(Move(SQ_E8, SQ_C8, MT_CASTLING));
    }
  }
}

void generate_legal(const Board& b, Movelist& out) {
  out.count = 0;
  Color us = b.side;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = b.piece[sq];
    if (pc == NO_PIECE || color_of(pc) != us) continue;
    add_piece_moves(b, sq, pc, out);
  }
  add_castling(b, out);

  int n = out.count;
  int w = 0;
  for (int i = 0; i < n; ++i) {
    if (b.is_legal(out.moves[i])) out.moves[w++] = out.moves[i];
  }
  out.count = w;
}

}  // namespace cortex
