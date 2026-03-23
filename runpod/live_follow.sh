#!/usr/bin/env bash
set -euo pipefail

# Live status viewer for current RunPod training run.
# Usage:
#   ./runpod/live_follow.sh
# Optional env:
#   RUNPOD_HOST=67.223.143.80 RUNPOD_PORT=18776 SSH_KEY=~/.ssh/id_ed25519 INTERVAL=5 ./runpod/live_follow.sh

RUNPOD_HOST="${RUNPOD_HOST:-67.223.143.80}"
RUNPOD_PORT="${RUNPOD_PORT:-18776}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${INTERVAL:-5}"

OUT_DIR="/workspace/Chess/runpod/out-full"
TRAIN_LOG="$OUT_DIR/launcher-maxeff4.log"
CKPT_DIR="$OUT_DIR/checkpoints"
WATCH_LOG="$OUT_DIR/watchdog-100s.log"

remote_status() {
  ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_HOST" "python3 - <<'PY'
from pathlib import Path
import re
import subprocess

out_dir = Path('/workspace/Chess/runpod/out-full')
train_log = out_dir / 'launcher-maxeff4.log'
ckpt_dir = out_dir / 'checkpoints'
watch_log = out_dir / 'watchdog-100s.log'

epoch_line = 'none'
step_line = 'none'
if train_log.exists():
    lines = train_log.read_text(errors='ignore').splitlines()
    # Prefer main training lines and ignore tiny verify trainer runs (e.g. samples=3).
    epoch_candidates = [ln for ln in lines if '[train] epoch=' in ln and '/120' in ln]
    step_candidates = [ln for ln in lines if '[train-step]' in ln and 'samples=' in ln]
    if epoch_candidates:
        epoch_line = epoch_candidates[-1]
    elif any('[train] epoch=' in ln for ln in lines):
        for ln in lines:
            if '[train] epoch=' in ln:
                epoch_line = ln
    # Keep only non-trivial steps if possible.
    big_steps = []
    for ln in step_candidates:
        m = re.search(r'samples=(\\d+)', ln)
        if m and int(m.group(1)) >= 1000:
            big_steps.append(ln)
    if big_steps:
        step_line = big_steps[-1]
    elif step_candidates:
        step_line = step_candidates[-1]

latest_ck = 'none'
if ckpt_dir.exists():
    cps = sorted(ckpt_dir.glob('epoch-*.pt'))
    if cps:
        latest_ck = cps[-1].name

watch_tail = 'none'
if watch_log.exists():
    wl = watch_log.read_text(errors='ignore').splitlines()
    if wl:
        watch_tail = wl[-1]

gpu = 'n/a'
gpu_p = 'none'
try:
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
         '--format=csv,noheader,nounits'],
        text=True
    ).strip()
except Exception:
    pass
try:
    gp = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        text=True
    ).strip()
    if gp:
        gpu_p = gp
except Exception:
    pass

train_proc = 'down'
try:
    out = subprocess.check_output(['pgrep', '-af', '/workspace/Chess/trainer/train_nnue.py'], text=True).strip()
    if out:
        train_proc = 'up'
except Exception:
    pass

print('train_proc=' + train_proc)
print('latest_checkpoint=' + latest_ck)
print('epoch_line=' + epoch_line)
print('step_line=' + step_line)
print('gpu=' + gpu)
print('gpu_procs=' + gpu_p)
print('watchdog=' + watch_tail)
PY"
}

progress_bar() {
  local pct="$1"
  local width=30
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  printf "["
  local i
  for ((i=0; i<filled; i++)); do printf "#"; done
  for ((i=0; i<empty; i++)); do printf "."; done
  printf "]"
}

while true; do
  status="$(remote_status || true)"
  train_proc="$(printf '%s\n' "$status" | awk -F= '/^train_proc=/{print $2}')"
  latest_ck="$(printf '%s\n' "$status" | awk -F= '/^latest_checkpoint=/{print $2}')"
  epoch_line="$(printf '%s\n' "$status" | sed -n 's/^epoch_line=//p')"
  step_line="$(printf '%s\n' "$status" | sed -n 's/^step_line=//p')"
  gpu_line="$(printf '%s\n' "$status" | sed -n 's/^gpu=//p')"
  gpu_procs="$(printf '%s\n' "$status" | sed -n 's/^gpu_procs=//p')"
  watchdog_line="$(printf '%s\n' "$status" | sed -n 's/^watchdog=//p')"

  ep_cur="?"
  ep_total="?"
  ep_pct="0"
  ep_eta="?"
  ep_loss="?"
  if [[ "$epoch_line" =~ epoch=([0-9]+)/([0-9]+).*loss=([0-9.]+).*eta=([0-9]+)s ]]; then
    ep_cur="${BASH_REMATCH[1]}"
    ep_total="${BASH_REMATCH[2]}"
    ep_loss="${BASH_REMATCH[3]}"
    ep_eta="${BASH_REMATCH[4]}s"
    ep_pct=$(( 100 * ep_cur / ep_total ))
  fi

  step_rate="?"
  if [[ "$step_line" =~ steps_per_s=([0-9.]+) ]]; then
    step_rate="${BASH_REMATCH[1]}"
  fi

  gpu_util="?"
  gpu_mem="?"
  gpu_pow="?"
  if [[ "$gpu_line" =~ ^([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9.]+)$ ]]; then
    gpu_util="${BASH_REMATCH[1]}%"
    gpu_mem="${BASH_REMATCH[2]}/${BASH_REMATCH[3]} MB"
    gpu_pow="${BASH_REMATCH[4]} W"
  fi

  printf '\033[2J\033[H'
  echo "----- $(date -u +%Y-%m-%dT%H:%M:%SZ) -----"
  echo "RunPod Training Dashboard"
  echo "Host: $RUNPOD_HOST  Port: $RUNPOD_PORT  Refresh: ${INTERVAL}s"
  echo
  echo "Run:       train_proc=$train_proc   checkpoint=$latest_ck"
  echo "Progress:  epoch $ep_cur/$ep_total  loss=$ep_loss  eta=$ep_eta  step/s=$step_rate"
  echo "           $(progress_bar "$ep_pct") ${ep_pct}%"
  echo "GPU:       util=$gpu_util  mem=$gpu_mem  power=$gpu_pow"
  echo "GPU PIDs:  $gpu_procs"
  echo "Watchdog:  $watchdog_line"
  echo "Epoch log: $epoch_line"
  echo
  sleep "$INTERVAL"
done

