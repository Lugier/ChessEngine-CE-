#!/usr/bin/env bash
# Wall-clock UCI search from startpos (no NNUE). For rough regression checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE="${ROOT}/engine/cortex"
DEPTH="${1:-9}"
if [[ ! -x "$EXE" ]]; then
  echo "Build first: $EXE missing" >&2
  exit 1
fi
export LC_ALL=C
/usr/bin/time -p sh -c "printf '%s\n' \
  uci \
  isready \
  'setoption name UseNNUE value false' \
  position startpos \
  'go depth ${DEPTH}' \
  quit | '$EXE' >/dev/null"
