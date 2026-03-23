#include "uci.hpp"
#include "board.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "search.hpp"
#include "tt.hpp"
#include <iostream>
#include <sstream>
#include <string>

namespace cortex {

static std::string square_str(Square s) {
  if (!is_ok(s)) return "-";
  char f = char('a' + file_of(s));
  char r = char('1' + rank_of(s));
  return std::string{f, r};
}

static std::string move_to_uci(Move m) {
  std::string s = square_str(m.from_sq()) + square_str(m.to_sq());
  if (m.type() == MT_PROMOTION) {
    const char* p = "nbrq";
    s += p[int(m.promo_kind())];
  }
  return s;
}

static Move uci_to_move(Board& b, const std::string& u) {
  if (u.size() < 4) return Move{};
  auto parse_sq = [](const std::string& w, size_t i) -> Square {
    int f = w[i] - 'a';
    int r = w[i + 1] - '1';
    if (f < 0 || f > 7 || r < 0 || r > 7) return SQ_NONE;
    return make_square(f, r);
  };
  Square from = parse_sq(u, 0);
  Square to = parse_sq(u, 2);
  if (!is_ok(from) || !is_ok(to)) return Move{};
  Piece moving = b.piece[from];
  if (moving == NO_PIECE) return Move{};
  if (type_of(moving) == PAWN) {
    int tr = rank_of(to);
    if (tr == 0 || tr == 7) {
      PromoKind pk = PROMO_QUEEN;
      if (u.size() >= 5) {
        char c = u[4];
        if (c == 'n') pk = PROMO_KNIGHT;
        else if (c == 'b')
          pk = PROMO_BISHOP;
        else if (c == 'r')
          pk = PROMO_ROOK;
      }
      return Move(from, to, MT_PROMOTION, pk);
    }
  }
  if (type_of(moving) == KING && std::abs(int(to) - int(from)) == 2)
    return Move(from, to, MT_CASTLING);
  if (type_of(moving) == PAWN && to == b.ep_square)
    return Move(from, to, MT_EN_PASSANT);
  return Move(from, to, MT_NORMAL);
}

static bool parse_position(Board& b, std::istringstream& is) {
  std::string tok;
  is >> tok;
  if (tok == "startpos") {
    if (!b.set_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
      return false;
  } else if (tok == "fen") {
    std::string fen, part;
    while (is >> part && part != "moves") {
      if (!fen.empty()) fen += ' ';
      fen += part;
    }
    if (!b.set_fen(fen)) return false;
    if (part == "moves") {
      std::string m;
      while (is >> m) {
        Move mv = uci_to_move(b, m);
        if (!mv) continue;
        UndoInfo u;
        b.do_move(mv, u);
        b.game_rep_keys.push_back(b.hash);
      }
    }
    return true;
  } else
    return false;

  is >> tok;
  if (tok == "moves") {
    std::string m;
    while (is >> m) {
      Move mv = uci_to_move(b, m);
      if (!mv) continue;
      UndoInfo u;
      b.do_move(mv, u);
      b.game_rep_keys.push_back(b.hash);
    }
  }
  return true;
}

void uci_loop() {
  Board board;
  board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  TranspositionTable tt(16);
  int hash_mb = 16;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    if (cmd.empty()) continue;
    if (cmd == "uci") {
      std::cout << "id name Cortex\n";
      std::cout << "id author local\n";
      std::cout << "option name Hash type spin default 16 min 1 max 512\n";
      std::cout << "option name UseNNUE type check default true\n";
      std::cout << "option name EvalFile type string default cortex.nnue\n";
      std::cout << "uciok\n" << std::flush;
    } else if (cmd == "isready") {
      std::cout << "readyok\n" << std::flush;
    } else if (cmd == "setoption") {
      std::string n, name, v, val;
      is >> n >> name;
      if (name == "Hash") {
        is >> v >> val;
        hash_mb = std::stoi(val);
        tt = TranspositionTable(size_t(hash_mb));
      } else if (name == "UseNNUE") {
        is >> v >> val;
        if (val == "false") g_nnue = NnueEvaluator{};
      } else if (name == "EvalFile") {
        is >> v;
        std::getline(is, val);
        if (!val.empty() && val[0] == ' ') val.erase(0, 1);
        if (!val.empty()) g_nnue.load_file(val);
      }
    } else if (cmd == "ucinewgame") {
      tt.clear();
    } else if (cmd == "position") {
      Board nb;
      if (!parse_position(nb, is)) continue;
      board = nb;
    } else if (cmd == "go") {
      SearchLimits lim;
      std::string w;
      while (is >> w) {
        if (w == "depth")
          is >> lim.depth;
        else if (w == "movetime")
          is >> lim.movetime_ms;
        else if (w == "nodes")
          is >> lim.nodes_max;
      }
      Move m = search_bestmove(board, lim, tt);
      std::cout << "bestmove " << move_to_uci(m) << "\n" << std::flush;
    } else if (cmd == "stop") {
      search_set_stop(true);
    } else if (cmd == "ponderhit" || cmd == "debug") {
      // no-op; keeps GUIs / tournament tools happy
    } else if (cmd == "quit")
      break;
  }
}

}  // namespace cortex
