#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace cortex {

struct Board {
  Piece piece[SQUARE_NB]{};
  Color side = WHITE;
  uint8_t castling = 0;
  Square ep_square = SQ_NONE;
  int halfmove = 0;
  int fullmove = 1;
  uint64_t hash = 0;

  static constexpr uint8_t CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4,
                           CASTLE_BQ = 8;

  Board() { clear(); }

  void clear();
  bool set_fen(const std::string& fen);
  std::string fen() const;

  Bitboard occupied() const;
  Bitboard occupied(Color c) const;
  Square king_sq(Color c) const;

  bool is_attacked(Square s, Color by) const;
  bool in_check() const { return is_attacked(king_sq(side), ~side); }

  void do_move(Move m, UndoInfo& u);
  void undo_move(Move m, const UndoInfo& u);

  bool is_legal(Move m) const;
};

}  // namespace cortex
