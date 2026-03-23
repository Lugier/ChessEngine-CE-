#!/usr/bin/env bash
set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:-67.223.143.80}"
RUNPOD_PORT="${RUNPOD_PORT:-18811}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${INTERVAL:-2}"
LOG_PATH="${LOG_PATH:-/workspace/Chess/runpod/out-4h/launcher-step2.log}"

ssh_cmd() {
  ssh -o StrictHostKeyChecking=accept-new -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_HOST" "$@"
}

fetch_status() {
  ssh_cmd "python3 - <<'PY'
from pathlib import Path
import re
import subprocess

log = Path('$LOG_PATH')
if not log.exists():
    print('exists=0')
    raise SystemExit(0)
print('exists=1')

lines = log.read_text(errors='replace').splitlines()
stage = 'unknown'
started = ''
score = ''
finished = 0
current = 0
total = 0
for ln in lines:
    if '[stage] selfplay generation' in ln:
        stage = 'selfplay'
    elif '[stage] pgn -> selfplay binpack' in ln:
        stage = 'pgn_to_binpack'
    elif '[stage] 4h finetune' in ln:
        stage = 'finetune'
    elif '[stage] quick sprt gate' in ln:
        stage = 'sprt'
    elif '== 4h selfplay run done' in ln:
        stage = 'done'
    m = re.search(r'Started game\\s+(\\d+)\\s+of\\s+(\\d+)', ln)
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        started = ln.strip()
    if 'Finished game ' in ln:
        finished += 1
    if 'Score of base vs cand:' in ln:
        score = ln.strip()

gpu = 'n/a'
try:
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
         '--format=csv,noheader,nounits'],
        text=True
    ).strip()
except Exception:
    pass

print(f'stage={stage}')
print(f'current={current}')
print(f'total={total}')
print(f'finished={finished}')
print(f'started={started}')
print(f'score={score}')
print(f'gpu={gpu}')
print(f'mtime={int(log.stat().st_mtime)}')
PY"
}

val() {
  local key="$1"
  local src="$2"
  printf '%s\n' "$src" | awk -v k="$key" '
    index($0, k "=") == 1 {
      sub("^" k "=", "", $0)
      print $0
      exit
    }'
}

fit() {
  local s="${1:-}"
  local w="${2:-70}"
  if (( ${#s} > w )); then
    printf "%s…" "${s:0:w-1}"
  else
    printf "%s" "$s"
  fi
}

bar() {
  local pct="${1:-0}"
  local width=28
  local p=${pct%.*}
  [[ -z "$p" ]] && p=0
  (( p < 0 )) && p=0
  (( p > 100 )) && p=100
  local filled=$(( p * width / 100 ))
  local empty=$(( width - filled ))
  printf "["
  for ((i=0; i<filled; i++)); do printf "="; done
  for ((i=0; i<empty; i++)); do printf "."; done
  printf "] %3d%%" "$p"
}

fmt_eta() {
  local s="$1"
  if [[ -z "$s" || "$s" == "n/a" || "$s" -lt 0 ]]; then
    printf "n/a"
    return
  fi
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  local sec=$(( s % 60 ))
  printf "%02dh %02dm %02ds" "$h" "$m" "$sec"
}

prev_cur=0
prev_ts=0
smoothed_rate=0

printf '\033[2J\033[H\033[?25l'
trap 'printf "\033[?25h\n"' EXIT

while true; do
  now_ts=$(date +%s)
  raw="$(fetch_status || true)"

  exists="$(val exists "$raw")"
  stage="$(val stage "$raw")"
  current="$(val current "$raw")"
  total="$(val total "$raw")"
  finished="$(val finished "$raw")"
  started="$(val started "$raw")"
  score="$(val score "$raw")"
  gpu_line="$(val gpu "$raw")"
  mtime="$(val mtime "$raw")"

  [[ -z "$current" ]] && current=0
  [[ -z "$total" ]] && total=0
  [[ -z "$finished" ]] && finished=0

  pct="0.0"
  if (( total > 0 )); then
    pct=$(python3 - <<PY
c=$current
t=$total
print(f"{(100.0*c/max(1,t)):.1f}")
PY
)
  fi

  inst_rate=0
  if (( prev_ts > 0 && now_ts > prev_ts )); then
    dc=$(( current - prev_cur ))
    dt=$(( now_ts - prev_ts ))
    if (( dt > 0 && dc >= 0 )); then
      inst_rate=$(python3 - <<PY
dc=$dc
dt=$dt
print(f"{dc/dt:.3f}")
PY
)
    fi
  fi
  if [[ "$inst_rate" != "0" ]]; then
    if [[ "$smoothed_rate" == "0" ]]; then
      smoothed_rate="$inst_rate"
    else
      smoothed_rate=$(python3 - <<PY
a=float("$smoothed_rate")
b=float("$inst_rate")
print(f"{0.75*a + 0.25*b:.3f}")
PY
)
    fi
  fi
  prev_cur=$current
  prev_ts=$now_ts

  eta_s="n/a"
  if (( total > 0 )); then
    rem=$(( total - current ))
    eta_s=$(python3 - <<PY
rem=$rem
rate=float("$smoothed_rate")
if rate > 1e-6:
    print(int(rem / rate))
else:
    print(-1)
PY
)
  fi

  gpu_util="n/a"; gpu_mem="n/a"; gpu_pow="n/a"; gpu_temp="n/a"
  if [[ "$gpu_line" =~ ^([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9.]+),[[:space:]]*([0-9]+)$ ]]; then
    gpu_util="${BASH_REMATCH[1]}%"
    gpu_mem="${BASH_REMATCH[3]} MiB / ${BASH_REMATCH[4]} MiB"
    gpu_pow="${BASH_REMATCH[5]}W"
    gpu_temp="${BASH_REMATCH[6]}C"
  fi

  lag_s="n/a"
  if [[ -n "$mtime" ]]; then
    lag_s=$(( now_ts - mtime ))
  fi

  printf '\033[H'
  echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
  echo "║                         SELFPLAY LIVE DASHBOARD (NO FLICKER)                     ║"
  echo "╠══════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Time (UTC): %-70s ║\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "║ Host: %-75s ║\n" "$(fit "$RUNPOD_HOST:$RUNPOD_PORT | interval=${INTERVAL}s | log_lag=${lag_s}s" 75)"
  echo "╠══════════════════════════════════════════════════════════════════════════════════╣"
  if [[ "$exists" != "1" ]]; then
    printf "║ %-80s ║\n" "Log not found yet: $LOG_PATH"
    printf "║ %-80s ║\n" "Waiting for selfplay to start..."
  else
    printf "║ Stage           : %-63s ║\n" "$(fit "${stage:-unknown}" 63)"
    printf "║ Progress        : %-63s ║\n" "$(fit "${current}/${total} (finished lines: ${finished})" 63)"
    printf "║ Progress Bar    : %-63s ║\n" "$(fit "$(bar "$pct")" 63)"
    printf "║ Speed           : %-63s ║\n" "$(fit "${smoothed_rate} games/s (smoothed)" 63)"
    printf "║ ETA             : %-63s ║\n" "$(fit "$(fmt_eta "$eta_s")" 63)"
    printf "║ Last Started    : %-63s ║\n" "$(fit "${started:-n/a}" 63)"
    printf "║ Current Score   : %-63s ║\n" "$(fit "${score:-n/a}" 63)"
    printf "║ GPU             : %-63s ║\n" "$(fit "util=${gpu_util} | mem=${gpu_mem} | power=${gpu_pow} | temp=${gpu_temp}" 63)"
  fi
  echo "╚══════════════════════════════════════════════════════════════════════════════════╝"

  sleep "$INTERVAL"
done
