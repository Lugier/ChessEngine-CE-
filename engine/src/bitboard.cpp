// Precomputed attacks and geometry (lines, between) for movegen and SEE.
#include "bitboard.hpp"

namespace cortex {

Bitboard SQUARE_BB[SQUARE_NB];
Bitboard LINE_BB[SQUARE_NB][SQUARE_NB];
Bitboard BETWEEN_BB[SQUARE_NB][SQUARE_NB];
Bitboard PSEUDO_ROOK_ATTACKS[SQUARE_NB];
Bitboard PSEUDO_BISHOP_ATTACKS[SQUARE_NB];
Bitboard KNIGHT_ATTACKS[SQUARE_NB];
Bitboard KING_ATTACKS[SQUARE_NB];
Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];

static constexpr int ROOK_DIRS[4] = {1, -1, 8, -8};
static constexpr int BISHOP_DIRS[4] = {9, 7, -7, -9};

static Bitboard ray_attack(Square sq, int dir, Bitboard occupied) {
  Bitboard atk = 0;
  int f = file_of(sq);
  int r = rank_of(sq);
  for (;;) {
    int nf = f;
    int nr = r;
    if (dir == 1 || dir == 9 || dir == -7) nf++;
    if (dir == -1 || dir == -9 || dir == 7) nf--;
    if (dir == 8 || dir == 9 || dir == 7) nr++;
    if (dir == -8 || dir == -9 || dir == -7) nr--;
    if (nf < 0 || nf > 7 || nr < 0 || nr > 7) break;
    f = nf;
    r = nr;
    Square ts = make_square(f, r);
    atk |= square_bb(ts);
    if (occupied & square_bb(ts)) break;
  }
  return atk;
}

Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
  Bitboard r = 0;
  if (pt == ROOK || pt == QUEEN) {
    for (int d : ROOK_DIRS) r |= ray_attack(sq, d, occupied);
  }
  if (pt == BISHOP || pt == QUEEN) {
    for (int d : BISHOP_DIRS) r |= ray_attack(sq, d, occupied);
  }
  return r;
}

Bitboard attacks_from(Piece pc, Square sq, Bitboard occupied) {
  PieceType pt = type_of(pc);
  Color c = color_of(pc);
  switch (pt) {
    case PAWN:
      return PAWN_ATTACKS[c][sq];
    case KNIGHT:
      return KNIGHT_ATTACKS[sq];
    case BISHOP:
    case ROOK:
    case QUEEN:
      return sliding_attack(pt, sq, occupied);
    case KING:
      return KING_ATTACKS[sq];
    default:
      return 0;
  }
}

void init_bitboards() {
  for (int s = 0; s < SQUARE_NB; ++s) {
    SQUARE_BB[s] = 1ULL << s;
    KNIGHT_ATTACKS[s] = 0;
    KING_ATTACKS[s] = 0;
    PAWN_ATTACKS[WHITE][s] = 0;
    PAWN_ATTACKS[BLACK][s] = 0;
  }

  static const int KD[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2},
                               {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    int f = file_of(sq);
    int r = rank_of(sq);
    for (auto [df, dr] : KD) {
      int nf = f + df, nr = r + dr;
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
        KNIGHT_ATTACKS[sq] |= square_bb(make_square(nf, nr));
    }
    for (int df = -1; df <= 1; ++df)
      for (int dr = -1; dr <= 1; ++dr) {
        if (!df && !dr) continue;
        int nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
          KING_ATTACKS[sq] |= square_bb(make_square(nf, nr));
      }
    if (f > 0 && r < 7) PAWN_ATTACKS[WHITE][sq] |= square_bb(sq + 7);
    if (f < 7 && r < 7) PAWN_ATTACKS[WHITE][sq] |= square_bb(sq + 9);
    if (f > 0 && r > 0) PAWN_ATTACKS[BLACK][sq] |= square_bb(sq - 9);
    if (f < 7 && r > 0) PAWN_ATTACKS[BLACK][sq] |= square_bb(sq - 7);
  }

  for (Square s1 = SQ_A1; s1 <= SQ_H8; s1 = Square(int(s1) + 1)) {
    PSEUDO_ROOK_ATTACKS[s1] = sliding_attack(ROOK, s1, 0);
    PSEUDO_BISHOP_ATTACKS[s1] = sliding_attack(BISHOP, s1, 0);
    for (Square s2 = SQ_A1; s2 <= SQ_H8; s2 = Square(int(s2) + 1)) {
      LINE_BB[s1][s2] = 0;
      BETWEEN_BB[s1][s2] = 0;
      if (s1 == s2) continue;
      if ((PSEUDO_ROOK_ATTACKS[s1] & square_bb(s2)) &&
          !(PSEUDO_BISHOP_ATTACKS[s1] & square_bb(s2))) {
        for (int d : ROOK_DIRS) {
          Bitboard ray = ray_attack(s1, d, 0);
          if (ray & square_bb(s2)) {
            LINE_BB[s1][s2] = ray | square_bb(s1);
            Bitboard b = 0;
            Square t = s1;
            for (;;) {
              t = Square(int(t) + d);
              if (!is_ok(t)) break;
              if (t == s2) break;
              b |= square_bb(t);
            }
            BETWEEN_BB[s1][s2] = b;
            break;
          }
        }
      } else if ((PSEUDO_BISHOP_ATTACKS[s1] & square_bb(s2)) &&
                 !(PSEUDO_ROOK_ATTACKS[s1] & square_bb(s2))) {
        for (int d : BISHOP_DIRS) {
          Bitboard ray = ray_attack(s1, d, 0);
          if (ray & square_bb(s2)) {
            LINE_BB[s1][s2] = ray | square_bb(s1);
            Bitboard b = 0;
            Square t = s1;
            for (;;) {
              t = Square(int(t) + d);
              if (!is_ok(t)) break;
              if (t == s2) break;
              b |= square_bb(t);
            }
            BETWEEN_BB[s1][s2] = b;
            break;
          }
        }
      }
    }
  }
}

}  // namespace cortex
