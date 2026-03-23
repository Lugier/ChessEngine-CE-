#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUTECHESS="${CUTECHESS_BIN:-cutechess-cli}"
GAMES="${GAMES:-200}"
TC="${TC:-40/8+0.08}"
THREADS="${THREADS:-1}"
OPENINGS="${OPENINGS:-}"
OUT="${OUT:-$ROOT/runpod/out/match.pgn}"
BASE="${BASE_ENGINE:-$ROOT/engine/cortex}"
CAND="${CAND_ENGINE:-$ROOT/engine/cortex}"
BASE_NET="${BASE_NET:-}"
CAND_NET="${CAND_NET:-$ROOT/engine/cortex.nnue}"

if ! command -v "$CUTECHESS" >/dev/null 2>&1; then
  echo "cutechess-cli not found. Set CUTECHESS_BIN or install cutechess-cli." >&2
  exit 1
fi
if [[ ! -x "$BASE" ]]; then
  echo "base engine not executable: $BASE" >&2
  exit 1
fi
if [[ ! -x "$CAND" ]]; then
  echo "candidate engine not executable: $CAND" >&2
  exit 1
fi
if [[ -n "$CAND_NET" && ! -r "$CAND_NET" ]]; then
  echo "candidate net not readable: $CAND_NET" >&2
  exit 1
fi
if [[ -n "$BASE_NET" && ! -r "$BASE_NET" ]]; then
  echo "base net not readable: $BASE_NET" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
base_cmd="option.Hash=1024 option.Threads=$THREADS option.UseNNUE=false"
cand_cmd="option.Hash=1024 option.Threads=$THREADS option.UseNNUE=true"
if [[ -n "$BASE_NET" ]]; then
  base_cmd="$base_cmd option.EvalFile=$BASE_NET option.UseNNUE=true"
fi
if [[ -n "$CAND_NET" ]]; then
  cand_cmd="$cand_cmd option.EvalFile=$CAND_NET"
fi

OPEN_ARGS=()
if [[ -n "$OPENINGS" ]]; then
  if [[ ! -r "$OPENINGS" ]]; then
    echo "openings file not readable: $OPENINGS" >&2
    exit 1
  fi
  OPEN_ARGS=(-openings file="$OPENINGS" format=pgn order=random)
fi

SPRT_ARGS=()
if [[ "${SPRT_MODE:-0}" == "1" ]]; then
  SPRT_ARGS=(-sprt elo0="${ELO0:-0}" elo1="${ELO1:-5}" alpha="${ALPHA:-0.05}" beta="${BETA:-0.05}")
fi

"$CUTECHESS" \
  -engine name=base cmd="$BASE" proto=uci $base_cmd \
  -engine name=cand cmd="$CAND" proto=uci $cand_cmd \
  -each tc="$TC" \
  -games "$GAMES" -repeat -recover \
  "${SPRT_ARGS[@]}" \
  "${OPEN_ARGS[@]}" \
  -pgnout "$OUT"
