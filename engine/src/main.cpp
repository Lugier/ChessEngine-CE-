// Entry: optional `perft N` for movegen regression; else stdin UCI loop.
#include "bitboard.hpp"
#include "board.hpp"
#include "eval_classic.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "uci.hpp"
#include "zobrist.hpp"
#include <cstdlib>
#include <iostream>

namespace cortex {

static uint64_t perft(Board& b, int depth) {
  if (depth <= 0) return 1;
  Movelist ml;
  generate_legal(b, ml);
  uint64_t n = 0;
  for (int i = 0; i < ml.count; ++i) {
    UndoInfo u;
    b.do_move(ml.moves[i], u);
    n += perft(b, depth - 1);
    b.undo_move(ml.moves[i], u);
  }
  return n;
}

}  // namespace cortex

int main(int argc, char** argv) {
  cortex::init_bitboards();
  cortex::init_zobrist();

  if (argc >= 2 && std::string(argv[1]) == "perft") {
    int d = (argc >= 3) ? std::atoi(argv[2]) : 5;
    cortex::Board b;
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << cortex::perft(b, d) << std::endl;
    return 0;
  }

  if (argc >= 2 && std::string(argv[1]) == "eval") {
    cortex::Board b;
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << cortex::evaluate_classic(b) << std::endl;
    return 0;
  }

  cortex::g_nnue.load_file("cortex.nnue");

  cortex::uci_loop();
  return 0;
}
