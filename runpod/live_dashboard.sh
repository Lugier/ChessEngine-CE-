#!/usr/bin/env bash
set -euo pipefail

RUNPOD_HOST="${RUNPOD_HOST:-67.223.143.80}"
RUNPOD_PORT="${RUNPOD_PORT:-18811}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${INTERVAL:-4}"

ssh_cmd() {
  ssh -o StrictHostKeyChecking=accept-new -p "$RUNPOD_PORT" -i "$SSH_KEY" "root@$RUNPOD_HOST" "$@"
}

remote_status() {
  ssh_cmd "python3 - <<'PY'
from pathlib import Path
import re
import subprocess
import time

root = Path('/workspace/Chess')
loop_log = root / 'runpod' / 'out-loop' / 'launcher.log'
warm_log = root / 'runpod' / 'out-warmstart' / 'training.log'
fourh_log = root / 'runpod' / 'out-4h' / 'train-4h.log'

now = time.time()

def tail(path: Path, n: int = 500):
    if not path.exists():
        return []
    lines = path.read_text(errors='replace').splitlines()
    return lines[-n:]

def latest(lines, needle):
    out = ''
    for ln in lines:
        if needle in ln:
            out = ln
    return out

def parse_epoch(line):
    if not line:
        return None
    m = re.search(r'epoch=(\\d+)/(\\d+).*eta=([0-9]+)s', line)
    if not m:
        return None
    cur, total, eta = map(int, m.groups())
    pct = 100.0 * cur / max(1, total)
    return cur, total, pct, eta

def parse_step(line):
    if not line:
        return None
    m = re.search(r'step=(\\d+)/(\\d+).*\\(([0-9.]+)%\\).*steps_per_s=([0-9.]+)', line)
    if not m:
        return None
    cur, total = int(m.group(1)), int(m.group(2))
    pct = float(m.group(3))
    sps = float(m.group(4))
    return cur, total, pct, sps

def parse_selfplay(lines):
    stage = ''
    progress = ''
    score = ''
    for ln in lines:
        if '[stage] selfplay generation' in ln:
            stage = 'selfplay_generation'
        elif '[stage] pgn -> selfplay binpack' in ln:
            stage = 'pgn_to_binpack'
        elif '[stage] 4h finetune' in ln:
            stage = 'selfplay_finetune'
        elif '[stage] quick sprt gate' in ln:
            stage = 'sprt_gate'
        elif '== 4h selfplay run done' in ln:
            stage = 'cycle_done'
        m = re.search(r'Started game\\s+(\\d+)\\s+of\\s+(\\d+)', ln)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            progress = f'{cur}/{total}|{100.0*cur/max(1,total):.1f}'
        if 'Score of base vs cand:' in ln:
            score = ln.strip()
    return stage, progress, score

loop_lines = tail(loop_log, 1200)
warm_lines = tail(warm_log, 400)
fourh_lines = tail(fourh_log, 1200)

phase = 'idle'
if any('[phase] warmstart on static mixed data' in ln for ln in loop_lines):
    phase = 'warmstart'
if any('[phase] selfplay cycle=' in ln for ln in loop_lines):
    phase = 'selfplay'

stage = ''
if phase == 'warmstart':
    if any('[stage] 3/3 train_nnue start' in ln for ln in loop_lines):
        stage = 'warmstart_training'
    elif any('[stage] 2/3 preflight verify' in ln for ln in loop_lines):
        stage = 'warmstart_verify'
    elif any('[stage] 1/3 dataprep:' in ln for ln in loop_lines):
        stage = 'warmstart_dataprep'
    else:
        stage = 'warmstart_setup'
elif phase == 'selfplay':
    st, sp, sc = parse_selfplay(fourh_lines)
    stage = st or 'selfplay_setup'
else:
    stage = 'idle'

epoch_line = latest(warm_lines + loop_lines, '[train] epoch=')
step_line = latest(warm_lines + loop_lines, '[train-step]')
epoch = parse_epoch(epoch_line)
step = parse_step(step_line)

selfplay_stage, selfplay_progress, selfplay_score = parse_selfplay(fourh_lines)

next_stages = []
if stage.startswith('warmstart'):
    next_stages = ['selfplay_generation', 'pgn_to_binpack', 'selfplay_finetune', 'sprt_gate', 'promotion']
elif stage in ('selfplay_generation', 'pgn_to_binpack', 'selfplay_finetune', 'sprt_gate'):
    order = ['selfplay_generation', 'pgn_to_binpack', 'selfplay_finetune', 'sprt_gate', 'promotion']
    i = order.index(stage)
    next_stages = order[i+1:]
else:
    next_stages = ['warmstart', 'selfplay_generation', 'selfplay_finetune', 'sprt_gate']

gpu = ''
try:
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
         '--format=csv,noheader,nounits'],
        text=True
    ).strip()
except Exception:
    gpu = 'n/a'

proc_keys = ['continuous_loop.sh', 'train_30h.sh', 'train_4h_selfplay.sh', 'trainer/train_nnue.py', 'cutechess-cli']
procs = []
try:
    out = subprocess.check_output(['ps', '-eo', 'pid,pcpu,pmem,cmd'], text=True)
    for ln in out.splitlines()[1:]:
        if any(k in ln for k in proc_keys):
            procs.append(ln.strip())
except Exception:
    pass

last_error = ''
err_markers = ('Traceback', 'RuntimeError', 'ModuleNotFoundError', 'Killed', 'ERROR', 'Exception')
for ln in reversed(loop_lines[-500:] + fourh_lines[-500:] + warm_lines[-500:]):
    if any(m in ln for m in err_markers):
        last_error = ln.strip()
        break

health = 'ok'
if last_error:
    health = 'warning'
if not procs:
    health = 'stopped'

loop_age_s = ''
if loop_log.exists():
    loop_age_s = str(int(max(0.0, now - loop_log.stat().st_mtime)))
else:
    loop_age_s = 'n/a'

print(f'phase={phase}')
print(f'stage={stage}')
if epoch:
    print(f'epoch={epoch[0]}/{epoch[1]}')
    print(f'epoch_pct={epoch[2]:.1f}')
    print(f'epoch_eta_s={epoch[3]}')
else:
    print('epoch=')
    print('epoch_pct=')
    print('epoch_eta_s=')
if step:
    print(f'step={step[0]}/{step[1]}')
    print(f'step_pct={step[2]:.1f}')
    print(f'steps_per_s={step[3]:.2f}')
else:
    print('step=')
    print('step_pct=')
    print('steps_per_s=')
print(f'selfplay_progress={selfplay_progress}')
print(f'selfplay_score={selfplay_score}')
print(f'gpu={gpu}')
print(f'health={health}')
print(f'loop_age_s={loop_age_s}')
print('next=' + ','.join(next_stages))
print(f'last_error={last_error}')
print(f'procs={len(procs)}')
for i, p in enumerate(procs[:8], start=1):
    print(f'proc{i}={p}')
PY"
}

get_val() {
  local k="$1"
  local src="$2"
  printf '%s\n' "$src" | awk -v key="$k" '
    index($0, key "=") == 1 {
      sub("^" key "=", "", $0)
      print $0
      exit
    }'
}

bar() {
  local pct="${1:-0}"
  local width=36
  local p=${pct%.*}
  if [[ -z "$p" ]]; then p=0; fi
  if (( p < 0 )); then p=0; fi
  if (( p > 100 )); then p=100; fi
  local filled=$(( p * width / 100 ))
  local empty=$(( width - filled ))
  printf "["
  for ((i=0; i<filled; i++)); do printf "█"; done
  for ((i=0; i<empty; i++)); do printf "·"; done
  printf "] %3d%%" "$p"
}

fmt_eta() {
  local s="$1"
  if [[ -z "$s" ]]; then
    printf "n/a"
    return
  fi
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  local sec=$(( s % 60 ))
  printf "%02dh %02dm %02ds" "$h" "$m" "$sec"
}

while true; do
  raw="$(remote_status || true)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  phase="$(get_val phase "$raw")"
  stage="$(get_val stage "$raw")"
  epoch="$(get_val epoch "$raw")"
  epoch_pct="$(get_val epoch_pct "$raw")"
  epoch_eta_s="$(get_val epoch_eta_s "$raw")"
  step="$(get_val step "$raw")"
  step_pct="$(get_val step_pct "$raw")"
  steps_per_s="$(get_val steps_per_s "$raw")"
  selfplay_progress="$(get_val selfplay_progress "$raw")"
  selfplay_score="$(get_val selfplay_score "$raw")"
  gpu_line="$(get_val gpu "$raw")"
  health="$(get_val health "$raw")"
  loop_age_s="$(get_val loop_age_s "$raw")"
  next="$(get_val next "$raw")"
  last_error="$(get_val last_error "$raw")"
  procs="$(get_val procs "$raw")"

  gpu_util="n/a"; gpu_mem="n/a"; gpu_pow="n/a"; gpu_temp="n/a"
  if [[ "$gpu_line" =~ ^([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9]+),[[:space:]]*([0-9.]+)$ ]]; then
    gpu_util="${BASH_REMATCH[1]}%"
    gpu_mem="${BASH_REMATCH[3]} MiB / ${BASH_REMATCH[4]} MiB"
    gpu_temp="${BASH_REMATCH[5]}C"
    gpu_pow="${BASH_REMATCH[6]}W"
  fi

  health_icon="●"
  case "$health" in
    ok) health_icon="●";;
    warning) health_icon="▲";;
    stopped) health_icon="■";;
  esac

  printf '\033[2J\033[H'
  echo "╔══════════════════════════════════════════════════════════════════════════════════════╗"
  echo "║                              CHESS RUNPOD LIVE DASHBOARD                             ║"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Time (UTC): %-72s ║\n" "$ts"
  printf "║ Host: %-20s Port: %-7s Refresh: %-5ss Health: %-14s ║\n" "$RUNPOD_HOST" "$RUNPOD_PORT" "$INTERVAL" "$health_icon $health"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Current Phase : %-68s ║\n" "${phase:-n/a}"
  printf "║ Current Stage : %-68s ║\n" "${stage:-n/a}"
  printf "║ Next Stages   : %-68s ║\n" "${next:-n/a}"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Warmstart Epoch: %-67s ║\n" "${epoch:-n/a}"
  printf "║ Epoch Progress : %-67s ║\n" "$(bar "${epoch_pct:-0}")"
  printf "║ Step Progress  : %-67s ║\n" "$(bar "${step_pct:-0}")"
  printf "║ Throughput     : steps/s=%-10s ETA=%-45s ║\n" "${steps_per_s:-n/a}" "$(fmt_eta "${epoch_eta_s:-}")"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Selfplay       : %-68s ║\n" "${selfplay_progress:-n/a}"
  printf "║ Selfplay Score : %-68s ║\n" "${selfplay_score:-n/a}"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ GPU Util       : %-68s ║\n" "$gpu_util"
  printf "║ GPU VRAM       : %-68s ║\n" "$gpu_mem"
  printf "║ GPU Power/Temp : %-68s ║\n" "$gpu_pow / $gpu_temp"
  printf "║ Active Procs   : %-68s ║\n" "${procs:-0}"
  printf "║ Log Age        : %-68s ║\n" "${loop_age_s:-n/a}s"
  echo "╠══════════════════════════════════════════════════════════════════════════════════════╣"
  printf "║ Last Error/Alert: %-66s ║\n" "${last_error:-none}"
  echo "╚══════════════════════════════════════════════════════════════════════════════════════╝"

  i=1
  while [[ $i -le 4 ]]; do
    p="$(get_val "proc${i}" "$raw")"
    if [[ -n "$p" ]]; then
      printf "   • %s\n" "$p"
    fi
    i=$((i + 1))
  done

  sleep "$INTERVAL"
done
