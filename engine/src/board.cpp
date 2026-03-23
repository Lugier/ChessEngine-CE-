// Board state: FEN I/O, do/undo, Zobrist; game_rep_keys for repetition seed.
#include "board.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"
#include <cctype>
#include <sstream>

namespace cortex {

void Board::clear() {
  for (auto& p : piece) p = NO_PIECE;
  side = WHITE;
  castling = 0;
  ep_square = SQ_NONE;
  halfmove = 0;
  fullmove = 1;
  hash = 0;
  game_rep_keys.clear();
}

static Piece char_to_piece(char c) {
  Color col = std::isupper(c) ? WHITE : BLACK;
  char u = char(std::tolower(c));
  PieceType pt = NO_PIECE_TYPE;
  switch (u) {
    case 'p': pt = PAWN; break;
    case 'n': pt = KNIGHT; break;
    case 'b': pt = BISHOP; break;
    case 'r': pt = ROOK; break;
    case 'q': pt = QUEEN; break;
    case 'k': pt = KING; break;
    default: return NO_PIECE;
  }
  return make_piece(col, pt);
}

bool Board::set_fen(const std::string& fen) {
  clear();
  std::istringstream iss(fen);
  std::string board_tok, side_tok, cast_tok, ep_tok;
  int hm = 0, fm = 1;
  if (!(iss >> board_tok >> side_tok)) return false;
  iss >> cast_tok >> ep_tok >> hm >> fm;
  if (cast_tok.empty()) cast_tok = "-";
  if (ep_tok.empty()) ep_tok = "-";

  int rank = 7, file = 0;
  for (char c : board_tok) {
    if (c == '/') {
      rank--;
      file = 0;
      continue;
    }
    if (c >= '1' && c <= '8') {
      file += c - '0';
      continue;
    }
    Square sq = make_square(file, rank);
    Piece pc = char_to_piece(c);
    if (!is_ok(pc) || !is_ok(sq)) return false;
    piece[sq] = pc;
    file++;
  }

  side = (side_tok == "b") ? BLACK : WHITE;
  castling = 0;
  for (char c : cast_tok) {
    if (c == 'K') castling |= CASTLE_WK;
    if (c == 'Q') castling |= CASTLE_WQ;
    if (c == 'k') castling |= CASTLE_BK;
    if (c == 'q') castling |= CASTLE_BQ;
  }
  ep_square = SQ_NONE;
  if (ep_tok != "-" && ep_tok.size() >= 2) {
    int f = ep_tok[0] - 'a';
    int r = ep_tok[1] - '1';
    if (f >= 0 && f < 8 && r >= 0 && r < 8) ep_square = make_square(f, r);
  }
  halfmove = hm;
  fullmove = fm;

  uint64_t h = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    if (piece[sq] != NO_PIECE) h ^= ZOBRIST.psq[piece[sq]][sq];
  }
  if (side == BLACK) h ^= ZOBRIST.side;
  h ^= ZOBRIST.castling[castling & 15];
  if (ep_square != SQ_NONE) h ^= ZOBRIST.ep[file_of(ep_square)];
  hash = h;
  game_rep_keys.clear();
  game_rep_keys.push_back(hash);
  return true;
}

std::string Board::fen() const {
  std::string s;
  auto piece_char = [](Piece pc) -> char {
    PieceType pt = type_of(pc);
    const char* u = "PNBRQK";
    if (pt < PAWN || pt > KING) return '?';
    char ch = u[int(pt)];
    return color_of(pc) == WHITE ? ch : char(std::tolower(ch));
  };
  for (int r = 7; r >= 0; --r) {
    int empty = 0;
    for (int f = 0; f < 8; ++f) {
      Square sq = make_square(f, r);
      Piece pc = piece[sq];
      if (pc == NO_PIECE) {
        empty++;
        continue;
      }
      if (empty) {
        s += char('0' + empty);
        empty = 0;
      }
      s += piece_char(pc);
    }
    if (empty) s += char('0' + empty);
    if (r) s += '/';
  }
  s += side == WHITE ? " w " : " b ";
  std::string cr;
  if (castling & CASTLE_WK) cr += 'K';
  if (castling & CASTLE_WQ) cr += 'Q';
  if (castling & CASTLE_BK) cr += 'k';
  if (castling & CASTLE_BQ) cr += 'q';
  s += cr.empty() ? "-" : cr;
  s += ' ';
  if (ep_square == SQ_NONE) s += '-';
  else {
    s += char('a' + file_of(ep_square));
    s += char('1' + rank_of(ep_square));
  }
  s += ' ';
  s += std::to_string(halfmove);
  s += ' ';
  s += std::to_string(fullmove);
  return s;
}

Bitboard Board::occupied() const {
  Bitboard o = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1))
    if (piece[sq] != NO_PIECE) o |= square_bb(sq);
  return o;
}

Bitboard Board::occupied(Color c) const {
  Bitboard o = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = piece[sq];
    if (pc != NO_PIECE && color_of(pc) == c) o |= square_bb(sq);
  }
  return o;
}

Square Board::king_sq(Color c) const {
  Piece k = make_piece(c, KING);
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1))
    if (piece[sq] == k) return sq;
  return SQ_NONE;
}

bool Board::is_attacked(Square s, Color by) const {
  Bitboard occ = occupied();
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    Piece pc = piece[sq];
    if (pc == NO_PIECE || color_of(pc) != by) continue;
    if (attacks_from(pc, sq, occ) & square_bb(s)) return true;
  }
  return false;
}

bool Board::is_material_draw() const {
  int wp = 0, wn = 0, wb = 0, wr = 0, wq = 0;
  int bp = 0, bn = 0, bb = 0, br = 0, bq = 0;
  for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(int(sq) + 1)) {
    switch (piece[sq]) {
      case W_PAWN:
        wp++;
        break;
      case W_KNIGHT:
        wn++;
        break;
      case W_BISHOP:
        wb++;
        break;
      case W_ROOK:
        wr++;
        break;
      case W_QUEEN:
        wq++;
        break;
      case B_PAWN:
        bp++;
        break;
      case B_KNIGHT:
        bn++;
        break;
      case B_BISHOP:
        bb++;
        break;
      case B_ROOK:
        br++;
        break;
      case B_QUEEN:
        bq++;
        break;
      default:
        break;
    }
  }
  if (wp || bp || wr || br || wq || bq) return false;
  int wm = wn + wb, bm = bn + bb;
  if (wm == 0 && bm == 0) return true;
  if (wm == 0 && bm == 1) return true;
  if (bm == 0 && wm == 1) return true;
  return false;
}

static PieceType promo_to_pt(PromoKind pk) {
  static const PieceType map[] = {KNIGHT, BISHOP, ROOK, QUEEN};
  return map[pk];
}

void Board::do_move(Move m, UndoInfo& u) {
  u.captured = NO_PIECE;
  u.castling = castling;
  u.ep_square = int8_t(ep_square);
  u.halfmove = halfmove;
  u.hash = hash;

  Square from = m.from_sq();
  Square to = m.to_sq();
  Piece moving = piece[from];
  PieceType mpt = type_of(moving);
  Color us = side;
  Color them = ~us;

  hash ^= ZOBRIST.psq[moving][from];
  if (piece[to] != NO_PIECE) {
    u.captured = piece[to];
    hash ^= ZOBRIST.psq[piece[to]][to];
  }

  hash ^= ZOBRIST.castling[castling & 15];
  if (ep_square != SQ_NONE) hash ^= ZOBRIST.ep[file_of(ep_square)];

  ep_square = SQ_NONE;

  if (m.type() == MT_CASTLING) {
    piece[from] = NO_PIECE;
    piece[to] = moving;
    hash ^= ZOBRIST.psq[moving][to];
    Square rook_from, rook_to;
    if (to > from) {
      rook_from = Square(int(from) + 3);
      rook_to = Square(int(from) + 1);
    } else {
      rook_from = Square(int(from) - 4);
      rook_to = Square(int(from) - 1);
    }
    Piece rook = piece[rook_from];
    hash ^= ZOBRIST.psq[rook][rook_from];
    piece[rook_from] = NO_PIECE;
    piece[rook_to] = rook;
    hash ^= ZOBRIST.psq[rook][rook_to];
  } else if (m.type() == MT_EN_PASSANT) {
    Square cap = Square(to + (us == WHITE ? -8 : 8));
    u.captured = piece[cap];
    hash ^= ZOBRIST.psq[piece[cap]][cap];
    piece[cap] = NO_PIECE;
    piece[from] = NO_PIECE;
    piece[to] = moving;
    hash ^= ZOBRIST.psq[moving][to];
  } else {
    piece[from] = NO_PIECE;
    if (m.type() == MT_PROMOTION) {
      Piece promoted = make_piece(us, promo_to_pt(m.promo_kind()));
      piece[to] = promoted;
      hash ^= ZOBRIST.psq[promoted][to];
    } else {
      piece[to] = moving;
      hash ^= ZOBRIST.psq[moving][to];
    }
  }

  if (mpt == PAWN) {
    halfmove = 0;
    if (std::abs(int(to) - int(from)) == 16) {
      Square mid = Square((int(from) + int(to)) / 2);
      if ((PAWN_ATTACKS[us][mid] & occupied(them))) ep_square = mid;
    }
  } else if (u.captured != NO_PIECE)
    halfmove = 0;
  else
    halfmove++;

  if (mpt == KING) {
    castling &= us == WHITE ? uint8_t(~(CASTLE_WK | CASTLE_WQ))
                            : uint8_t(~(CASTLE_BK | CASTLE_BQ));
  }
  if (from == SQ_H1 || to == SQ_H1) castling &= ~CASTLE_WK;
  if (from == SQ_A1 || to == SQ_A1) castling &= ~CASTLE_WQ;
  if (from == SQ_H8 || to == SQ_H8) castling &= ~CASTLE_BK;
  if (from == SQ_A8 || to == SQ_A8) castling &= ~CASTLE_BQ;

  if (us == BLACK) fullmove++;
  side = them;
  hash ^= ZOBRIST.side;
  hash ^= ZOBRIST.castling[castling & 15];
  if (ep_square != SQ_NONE) hash ^= ZOBRIST.ep[file_of(ep_square)];
}

void Board::undo_move(Move m, const UndoInfo& u) {
  side = ~side;
  Color us = side;
  Square from = m.from_sq();
  Square to = m.to_sq();

  hash = u.hash;
  castling = u.castling;
  ep_square = Square(u.ep_square);
  halfmove = u.halfmove;
  if (us == BLACK) fullmove--;

  if (m.type() == MT_CASTLING) {
    Piece king = make_piece(us, KING);
    Piece rook = make_piece(us, ROOK);
    piece[to] = NO_PIECE;
    piece[from] = king;
    Square rook_from, rook_to;
    if (to > from) {
      rook_from = Square(int(from) + 3);
      rook_to = Square(int(from) + 1);
    } else {
      rook_from = Square(int(from) - 4);
      rook_to = Square(int(from) - 1);
    }
    piece[rook_to] = NO_PIECE;
    piece[rook_from] = rook;
  } else if (m.type() == MT_EN_PASSANT) {
    piece[to] = NO_PIECE;
    piece[from] = make_piece(us, PAWN);
    Square cap = Square(to + (us == WHITE ? -8 : 8));
    piece[cap] = u.captured;
  } else if (m.type() == MT_PROMOTION) {
    piece[to] = u.captured;
    piece[from] = make_piece(us, PAWN);
  } else {
    Piece moved = piece[to];
    piece[to] = u.captured;
    piece[from] = moved;
  }
}

bool Board::is_legal(Move m) const {
  Board c = *this;
  UndoInfo u;
  c.do_move(m, u);
  Color us = ~c.side;
  Square ksq = c.king_sq(us);
  return ksq == SQ_NONE || !c.is_attacked(ksq, c.side);
}

}  // namespace cortex
