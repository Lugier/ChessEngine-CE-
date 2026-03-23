#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALPHA="${ALPHA:-0.05}"
BETA="${BETA:-0.05}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
MATCH_OUT="${MATCH_OUT:-$ROOT/runpod/out/match.pgn}"

OUT="$MATCH_OUT" SPRT_MODE=1 ALPHA="$ALPHA" BETA="$BETA" ELO0="$ELO0" ELO1="$ELO1" \
  "$ROOT/scripts/cutechess_match.sh"

if ! command -v bayeselo >/dev/null 2>&1; then
  echo "SPRT hint: install bayeselo for automated accept/reject."
  echo "Match generated at $MATCH_OUT"
  exit 0
fi

cat <<EOF | bayeselo
readpgn $MATCH_OUT
elo
mm
exactdist
ratings
x
EOF

echo "SPRT thresholds configured: alpha=$ALPHA beta=$BETA elo0=$ELO0 elo1=$ELO1"
echo "Interpretation: promote candidate only on clear positive evidence."
