#pragma once

#include <cstdint>
#include <string>

namespace cortex {

enum Color : int8_t { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
inline Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int8_t {
  NO_PIECE_TYPE = -1,
  PAWN = 0,
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,
  PIECE_TYPE_NB = 6
};

enum Piece : int8_t {
  NO_PIECE = 0,
  W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
  PIECE_NB = 15
};

constexpr Piece make_piece(Color c, PieceType pt) {
  return Piece((c == WHITE ? 1 : 9) + int(pt));
}
constexpr Color color_of(Piece pc) { return pc < B_PAWN ? WHITE : BLACK; }
constexpr PieceType type_of(Piece pc) {
  if (pc == NO_PIECE) return NO_PIECE_TYPE;
  if (int(pc) <= int(W_KING)) return PieceType(int(pc) - int(W_PAWN));
  return PieceType(int(pc) - int(B_PAWN));
}
constexpr bool is_ok(Piece pc) { return pc >= W_PAWN && pc <= B_KING; }

enum Square : int8_t {
  SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
  SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
  SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
  SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
  SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
  SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
  SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
  SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
  SQ_NONE = 64,
  SQUARE_NB = 64
};

constexpr Square make_square(int file, int rank) {
  return Square(rank * 8 + file);
}
constexpr int file_of(Square s) { return int(s) & 7; }
constexpr int rank_of(Square s) { return int(s) >> 3; }
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

constexpr Square operator+(Square s, int d) { return Square(int(s) + d); }
constexpr Square operator-(Square s, int d) { return Square(int(s) - d); }

using Bitboard = uint64_t;

enum MoveType : uint8_t {
  MT_NORMAL = 0,
  MT_PROMOTION,
  MT_EN_PASSANT,
  MT_CASTLING
};

enum PromoKind : uint8_t { PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN };

struct Move {
  uint32_t data = 0;

  Move() = default;
  Move(Square from, Square to, MoveType mt = MT_NORMAL,
       PromoKind pk = PROMO_QUEEN)
      : data(uint32_t(from) | (uint32_t(to) << 6) | (uint32_t(mt) << 12) |
             (uint32_t(pk) << 14)) {}

  Square from_sq() const { return Square(data & 63); }
  Square to_sq() const { return Square((data >> 6) & 63); }
  MoveType type() const { return MoveType((data >> 12) & 3); }
  PromoKind promo_kind() const { return PromoKind((data >> 14) & 3); }

  explicit operator bool() const { return data != 0; }
  bool operator==(Move o) const { return data == o.data; }
  bool operator!=(Move o) const { return data != o.data; }
};

struct UndoInfo {
  Piece captured;
  uint8_t castling;
  int8_t ep_square;
  int halfmove;
  uint64_t hash;
};

constexpr int MAX_MOVES = 256;

}  // namespace cortex
