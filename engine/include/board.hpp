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
  /// Position hashes along the actual game line (UCI); used to seed repetition
  /// detection in search. Not modified by internal search make/unmake.
  std::vector<uint64_t> game_rep_keys;

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

  /// FIDE-style automatic draw: no pawns/rooks/queens and K vs K or K vs single minor.
  bool is_material_draw() const;

  void do_move(Move m, UndoInfo& u);
  void undo_move(Move m, const UndoInfo& u);

  bool is_legal(Move m) const;
};

}  // namespace cortex
