#pragma once

#include "types.hpp"

namespace cortex {

extern Bitboard SQUARE_BB[SQUARE_NB];
extern Bitboard LINE_BB[SQUARE_NB][SQUARE_NB];
extern Bitboard BETWEEN_BB[SQUARE_NB][SQUARE_NB];
extern Bitboard PSEUDO_ROOK_ATTACKS[SQUARE_NB];
extern Bitboard PSEUDO_BISHOP_ATTACKS[SQUARE_NB];
extern Bitboard KNIGHT_ATTACKS[SQUARE_NB];
extern Bitboard KING_ATTACKS[SQUARE_NB];
extern Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];

void init_bitboards();

inline Bitboard square_bb(Square s) { return SQUARE_BB[s]; }
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline Square lsb(Bitboard b) { return Square(__builtin_ctzll(b)); }

inline Bitboard pop_lsb(Bitboard& b) {
  Square s = lsb(b);
  b &= b - 1;
  return square_bb(s);
}

Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied);
Bitboard attacks_from(Piece pc, Square sq, Bitboard occupied);

}  // namespace cortex
