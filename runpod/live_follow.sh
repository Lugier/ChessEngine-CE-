#!/usr/bin/env bash
set -euo pipefail

# Live status viewer for RunPod training runs (4h selfplay + 30h).

RUNPOD_HOST="${RUNPOD_HOST:-67.223.143.80}"
RUNPOD_PORT="${RUNPOD_PORT:-18776}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${INTERVAL:-5}"

remote_status() {
  ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_HOST" "python3 - <<'PY'
from pathlib import Path
import re
import subprocess

root = Path('/workspace/Chess')
out_full = root / 'runpod' / 'out-full'
out_4h = root / 'runpod' / 'out-4h'
watch_log = out_full / 'watchdog-100s.log'

def tail_line(path: Path, marker: str):
    if not path.exists():
        return 'none'
    lines = path.read_text(errors='ignore').splitlines()
    out = 'none'
    for ln in lines:
        if marker in ln:
            out = ln
    return out

latest_ck = 'none'
ckpt_dir = out_full / 'checkpoints'
if ckpt_dir.exists():
    cps = sorted(ckpt_dir.glob('epoch-*.pt'))
    if cps:
        latest_ck = cps[-1].name

stage = 'unknown'
selfplay_progress = 'n/a'
selfplay_score = 'n/a'
phase = 'idle'
train_epoch = 'n/a'
train_step = 'n/a'
last_error = 'none'

log_4h = out_4h / 'train-4h.log'
if log_4h.exists():
    lines4 = log_4h.read_text(errors='ignore').splitlines()
    for ln in lines4:
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
    for ln in lines4:
        m = re.search(r'Started game\\s+(\\d+)\\s+of\\s+(\\d+)', ln)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            selfplay_progress = f'{cur}/{tot} ({100.0*cur/max(1,tot):.1f}%)'
        if 'Score of base vs cand:' in ln:
            selfplay_score = ln.strip()
        if 'Traceback' in ln or 'ModuleNotFoundError' in ln or 'RuntimeError' in ln:
            last_error = ln.strip()

epoch_line = tail_line(out_full / 'training.log', '[train] epoch=')
step_line = tail_line(out_full / 'training.log', '[train-step]')
if epoch_line == 'none':
    epoch_line = tail_line(out_full / 'launcher-maxeff4.log', '[train] epoch=')
if step_line == 'none':
    step_line = tail_line(out_full / 'launcher-maxeff4.log', '[train-step]')

if stage in {'finetune', 'sprt', 'done'}:
    phase = stage
else:
    phase = stage if stage != 'unknown' else 'train_or_idle'

if epoch_line != 'none':
    train_epoch = epoch_line
if step_line != 'none':
    train_step = step_line

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
run4h_proc = 'down'
try:
    out = subprocess.check_output(['pgrep', '-af', '/workspace/Chess/runpod/train_4h_selfplay.sh'], text=True).strip()
    if out:
        run4h_proc = 'up'
except Exception:
    pass

print('train_proc=' + train_proc)
print('run4h_proc=' + run4h_proc)
print('latest_checkpoint=' + latest_ck)
print('phase=' + phase)
print('selfplay_progress=' + selfplay_progress)
print('selfplay_score=' + selfplay_score)
print('epoch_line=' + train_epoch)
print('step_line=' + train_step)
print('gpu=' + gpu)
print('gpu_procs=' + gpu_p)
print('watchdog=' + watch_tail)
print('last_error=' + last_error)
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
  run4h_proc="$(printf '%s\n' "$status" | awk -F= '/^run4h_proc=/{print $2}')"
  latest_ck="$(printf '%s\n' "$status" | awk -F= '/^latest_checkpoint=/{print $2}')"
  phase="$(printf '%s\n' "$status" | sed -n 's/^phase=//p')"
  selfplay_progress="$(printf '%s\n' "$status" | sed -n 's/^selfplay_progress=//p')"
  selfplay_score="$(printf '%s\n' "$status" | sed -n 's/^selfplay_score=//p')"
  epoch_line="$(printf '%s\n' "$status" | sed -n 's/^epoch_line=//p')"
  step_line="$(printf '%s\n' "$status" | sed -n 's/^step_line=//p')"
  gpu_line="$(printf '%s\n' "$status" | sed -n 's/^gpu=//p')"
  gpu_procs="$(printf '%s\n' "$status" | sed -n 's/^gpu_procs=//p')"
  watchdog_line="$(printf '%s\n' "$status" | sed -n 's/^watchdog=//p')"
  last_error="$(printf '%s\n' "$status" | sed -n 's/^last_error=//p')"

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
  echo "Run:       train_proc=$train_proc  4h_proc=$run4h_proc  phase=$phase  checkpoint=$latest_ck"
  echo "Selfplay:  $selfplay_progress"
  echo "Score:     $selfplay_score"
  echo "Progress:  epoch $ep_cur/$ep_total  loss=$ep_loss  eta=$ep_eta  step/s=$step_rate"
  echo "           $(progress_bar "$ep_pct") ${ep_pct}%"
  echo "GPU:       util=$gpu_util  mem=$gpu_mem  power=$gpu_pow"
  echo "GPU PIDs:  $gpu_procs"
  echo "Watchdog:  $watchdog_line"
  echo "Epoch log: $epoch_line"
  echo "Last err:  $last_error"
  echo
  sleep "$INTERVAL"
done

