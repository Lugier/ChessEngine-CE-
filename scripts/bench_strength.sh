#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="$ROOT/engine/cortex"
DEPTH="${1:-10}"
NET="${2:-$ROOT/engine/cortex.nnue}"

if [[ ! -x "$EXE" ]]; then
  echo "missing engine binary: $EXE" >&2
  exit 1
fi

echo "== bench_strength depth=$DEPTH =="
cat <<EOF | "$EXE"
uci
isready
setoption name EvalFile value $NET
setoption name UseNNUE value true
isready
position startpos
go depth $DEPTH
position fen r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2PB1N2/PP3PPP/RNBQ1RK1 w - - 0 8
go depth $DEPTH
position fen 4rrk1/1pp2pp1/p1n1b2p/3pP3/3P1B2/2P2N1P/PP3PP1/R2R2K1 w - - 0 20
go depth $DEPTH
quit
EOF
