#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "$ROOT/data/processed-full" ]]; then
    DATA_DIR="$ROOT/data/processed-full"
  else
    DATA_DIR="$ROOT/data/processed"
  fi
fi
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out}"
CKPT_DIR="${CKPT_DIR:-$OUT_DIR/checkpoints}"
LOG="$OUT_DIR/training.log"
EPOCHS="${EPOCHS:-120}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LR="${LR:-1e-3}"
FEATURE_MODE="${FEATURE_MODE:-legacy}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-3000000}"
VAL_MAX_ROWS="${VAL_MAX_ROWS:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VAL_EVERY="${VAL_EVERY:-10}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
LOSS_FN="${LOSS_FN:-kldiv}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.01}"
WDL_TEMPERATURE="${WDL_TEMPERATURE:-1.2}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
CPU_THREADS="${CPU_THREADS:-4}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1000000}"
MIN_VAL_ROWS="${MIN_VAL_ROWS:-100000}"
STRICT_FULL_RUN="${STRICT_FULL_RUN:-1}"
TRAIN_BIN="${TRAIN_BIN:-$DATA_DIR/train.binpack}"
VAL_BIN="${VAL_BIN:-$DATA_DIR/val.binpack}"
OUT_NET="${OUT_NET:-$OUT_DIR/cortex.nnue}"
PY="${PYTHON_BIN:-python3}"
DATA_PY="${DATA_PYTHON_BIN:-$PY}"
AUTO_PREP="${AUTO_PREP:-1}"
HF_REPO_ID="${HF_REPO_ID:-Lichess/chess-position-evaluations}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-$ROOT/data/raw/lichess-evals}"
PARQUET_GLOB="${PARQUET_GLOB:-$HF_LOCAL_DIR/data/*.parquet}"
SUMMARY_OUT="${SUMMARY_OUT:-$DATA_DIR/summary.json}"
MIN_DEPTH="${MIN_DEPTH:-16}"
CP_CAP="${CP_CAP:-1200}"
VAL_RATIO="${VAL_RATIO:-0.02}"
CP_SCALE="${CP_SCALE:-400}"
MAX_ROWS="${MAX_ROWS:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-200000}"
PREP_WORKERS="${PREP_WORKERS:-32}"
POSITION_POLICY="${POSITION_POLICY:-mixed}"
MIX_ALL="${MIX_ALL:-0.50}"
MIX_TACTICAL="${MIX_TACTICAL:-0.30}"
MIX_QUIET="${MIX_QUIET:-0.20}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-8}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-0}"
AUTO_POST_VERIFY="${AUTO_POST_VERIFY:-1}"
AUTO_POST_EVAL="${AUTO_POST_EVAL:-1}"
EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_TC="${EVAL_TC:-40/8+0.08}"
EVAL_THREADS="${EVAL_THREADS:-1}"
EVAL_OPENINGS="${EVAL_OPENINGS:-}"
SPRT_ALPHA="${SPRT_ALPHA:-0.05}"
SPRT_BETA="${SPRT_BETA:-0.05}"
SPRT_ELO0="${SPRT_ELO0:-0}"
SPRT_ELO1="${SPRT_ELO1:-5}"
MATCH_OUT="${MATCH_OUT:-$OUT_DIR/match.pgn}"
ELO_REPORT="${ELO_REPORT:-$OUT_DIR/elo_report.txt}"
RUN_LOCK="${RUN_LOCK:-$OUT_DIR/train.lock}"
RESUME_ARG=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARG=(--resume "$RESUME")
fi

mkdir -p "$OUT_DIR" "$CKPT_DIR"
mkdir -p "$DATA_DIR" "$HF_LOCAL_DIR"
echo "== train_30h: start $(date -u) ==" | tee -a "$LOG"
echo "train=$TRAIN_BIN val=$VAL_BIN out=$OUT_NET" | tee -a "$LOG"
echo "auto_prep=$AUTO_PREP parquet_glob=$PARQUET_GLOB" | tee -a "$LOG"
echo "feature_mode=$FEATURE_MODE epochs=$EPOCHS lr=$LR save_every=$SAVE_EVERY" | tee -a "$LOG"
echo "batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS" | tee -a "$LOG"
echo "max_train_rows=$MAX_TRAIN_ROWS val_max_rows=$VAL_MAX_ROWS" | tee -a "$LOG"
echo "log_interval=$LOG_INTERVAL" | tee -a "$LOG"
echo "prefetch_factor=$PREFETCH_FACTOR" | tee -a "$LOG"
echo "val_every=$VAL_EVERY" | tee -a "$LOG"
echo "optimizer=$OPTIMIZER scheduler=$SCHEDULER loss=$LOSS_FN ema=$USE_EMA amp_dtype=$AMP_DTYPE" | tee -a "$LOG"
echo "strict_full_run=$STRICT_FULL_RUN min_train_rows=$MIN_TRAIN_ROWS min_val_rows=$MIN_VAL_ROWS" | tee -a "$LOG"
echo "prep_workers=$PREP_WORKERS" | tee -a "$LOG"
echo "position_policy=$POSITION_POLICY mix_all=$MIX_ALL mix_tactical=$MIX_TACTICAL mix_quiet=$MIX_QUIET" | tee -a "$LOG"
echo "python_train=$PY python_data=$DATA_PY install_runtime_deps=$INSTALL_RUNTIME_DEPS" | tee -a "$LOG"
echo "auto_post_verify=$AUTO_POST_VERIFY auto_post_eval=$AUTO_POST_EVAL" | tee -a "$LOG"

acquire_lock() {
  mkdir -p "$(dirname "$RUN_LOCK")"
  if [[ -f "$RUN_LOCK" ]]; then
    local old_pid
    old_pid="$(<"$RUN_LOCK")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "another training run is active (pid=$old_pid, lock=$RUN_LOCK); refusing parallel start" >&2
      exit 1
    fi
    rm -f "$RUN_LOCK"
  fi
  echo "$$" > "$RUN_LOCK"
  trap 'rm -f "$RUN_LOCK"' EXIT
}

with_retries() {
  local n=1
  local max_n="$1"
  shift
  until "$@"; do
    if [[ "$n" -ge "$max_n" ]]; then
      return 1
    fi
    n=$((n + 1))
    echo "retrying ($n/$max_n) after ${RETRY_DELAY_SEC}s: $*" | tee -a "$LOG"
    sleep "$RETRY_DELAY_SEC"
  done
}

stage() {
  echo "[stage] $1 | $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
}

ensure_data_binpacks() {
  if [[ -f "$TRAIN_BIN" && -f "$VAL_BIN" ]]; then
    echo "binpacks already available; skipping dataprep" | tee -a "$LOG"
    return 0
  fi
  if [[ "$AUTO_PREP" != "1" ]]; then
    echo "missing binpacks and AUTO_PREP=0; aborting" >&2
    return 1
  fi

  if [[ "$INSTALL_RUNTIME_DEPS" == "1" ]]; then
    echo "binpacks missing; installing dataprep dependencies at runtime" | tee -a "$LOG"
    with_retries "$DOWNLOAD_RETRIES" "$DATA_PY" -m pip install -q --break-system-packages -r "$ROOT/data/requirements.txt" huggingface_hub
  else
    "$DATA_PY" - <<'PY'
from importlib.util import find_spec
mods = ("duckdb", "chess", "huggingface_hub")
missing = [m for m in mods if find_spec(m) is None]
if missing:
    raise SystemExit(
        "missing python modules: "
        + ", ".join(missing)
        + ". Build the RunPod image with dependencies or set INSTALL_RUNTIME_DEPS=1."
    )
PY
  fi

  if [[ -z "$(compgen -G "$PARQUET_GLOB" || true)" ]]; then
    echo "no parquet files found; downloading from Hugging Face dataset $HF_REPO_ID" | tee -a "$LOG"
    HF_REPO_ID="$HF_REPO_ID" HF_LOCAL_DIR="$HF_LOCAL_DIR" HF_TOKEN="${HF_TOKEN:-}" \
      with_retries "$DOWNLOAD_RETRIES" "$DATA_PY" - <<'PY'
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["HF_LOCAL_DIR"],
    allow_patterns=["*.parquet", "README.md", "dataset_infos.json"],
    token=os.getenv("HF_TOKEN") or None,
)
print("hf dataset download complete")
PY
  fi

  # Some datasets place parquet shards in HF_LOCAL_DIR/data, others directly in HF_LOCAL_DIR.
  if [[ -z "$(compgen -G "$PARQUET_GLOB" || true)" && -n "$(compgen -G "$HF_LOCAL_DIR/*.parquet" || true)" ]]; then
    PARQUET_GLOB="$HF_LOCAL_DIR/*.parquet"
    echo "using fallback parquet_glob=$PARQUET_GLOB" | tee -a "$LOG"
  fi
  if [[ -z "$(compgen -G "$PARQUET_GLOB" || true)" ]]; then
    echo "no parquet shards found after download in $HF_LOCAL_DIR" >&2
    return 1
  fi

  stage "1/3 dataprep: building train/val binpacks"
  "$DATA_PY" "$ROOT/data/prepare_lichess_quiet.py" \
    --parquet "$PARQUET_GLOB" \
    --train-out "$TRAIN_BIN" \
    --val-out "$VAL_BIN" \
    --summary-out "$SUMMARY_OUT" \
    --min-depth "$MIN_DEPTH" \
    --cp-cap "$CP_CAP" \
    --val-ratio "$VAL_RATIO" \
    --cp-scale "$CP_SCALE" \
    --max-rows "$MAX_ROWS" \
    --chunk-size "$CHUNK_SIZE" \
    --workers "$PREP_WORKERS" \
    --position-policy "$POSITION_POLICY" \
    --mix-all "$MIX_ALL" \
    --mix-tactical "$MIX_TACTICAL" \
    --mix-quiet "$MIX_QUIET" | tee -a "$LOG"
}

validate_dataset_selection() {
  if [[ "$STRICT_FULL_RUN" != "1" ]]; then
    return 0
  fi
  if [[ "$FEATURE_MODE" != "kingbucket" ]]; then
    echo "STRICT_FULL_RUN=1 requires FEATURE_MODE=kingbucket (got: $FEATURE_MODE)" >&2
    return 1
  fi
  if [[ "$DATA_DIR" != *"/processed-full"* ]]; then
    echo "STRICT_FULL_RUN=1 requires DATA_DIR to point to processed-full (got: $DATA_DIR)" >&2
    return 1
  fi
}

validate_binpack_sizes() {
  "$DATA_PY" - "$TRAIN_BIN" "$VAL_BIN" "$MIN_TRAIN_ROWS" "$MIN_VAL_ROWS" <<'PY'
import struct
import sys
from pathlib import Path

train = Path(sys.argv[1])
val = Path(sys.argv[2])
min_train = int(sys.argv[3])
min_val = int(sys.argv[4])

def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        b = f.read(4)
    if len(b) != 4:
        raise SystemExit(f"invalid binpack header: {path}")
    return struct.unpack("<I", b)[0]

tr = count_rows(train)
vr = count_rows(val)
print(f"binpack_rows train={tr} val={vr}")
if tr < min_train:
    raise SystemExit(f"train rows too small: {tr} < {min_train} ({train})")
if vr < min_val:
    raise SystemExit(f"val rows too small: {vr} < {min_val} ({val})")
PY
}

adapt_training_memory_profile() {
  # FenDataset keeps many strings in-memory; with many loader workers this can OOM.
  # Keep an aggressive but stable profile by capping rows/workers for this trainer.
  if [[ "$MAX_TRAIN_ROWS" -gt 300000 && "$NUM_WORKERS" -gt 24 ]]; then
    echo "adaptive-cap: reducing MAX_TRAIN_ROWS from $MAX_TRAIN_ROWS to 300000 for stability with NUM_WORKERS=$NUM_WORKERS" | tee -a "$LOG"
    MAX_TRAIN_ROWS=300000
  fi
  if [[ "$MAX_TRAIN_ROWS" -gt 600000 && "$NUM_WORKERS" -gt 16 ]]; then
    echo "adaptive-cap: reducing NUM_WORKERS from $NUM_WORKERS to 16 for stability with MAX_TRAIN_ROWS=$MAX_TRAIN_ROWS" | tee -a "$LOG"
    NUM_WORKERS=16
  fi
}

acquire_lock
validate_dataset_selection
ensure_data_binpacks

if [[ ! -f "$TRAIN_BIN" ]]; then
  echo "missing train binpack: $TRAIN_BIN" >&2
  exit 1
fi
if [[ ! -f "$VAL_BIN" && "${ALLOW_MISSING_VAL:-0}" != "1" ]]; then
  echo "missing val binpack: $VAL_BIN (set ALLOW_MISSING_VAL=1 to bypass)" >&2
  exit 1
fi
validate_binpack_sizes | tee -a "$LOG"
adapt_training_memory_profile

if [[ "${SKIP_PREFLIGHT_VERIFY:-0}" != "1" ]]; then
  stage "2/3 preflight verify"
  SKIP_TRAINER=1 "$ROOT/scripts/verify.sh"
fi

stage "3/3 train_nnue start"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
train_cmd=(
  "$PY" -u "$ROOT/trainer/train_nnue.py"
  --data "$TRAIN_BIN"
  --val-data "$VAL_BIN"
  --epochs "$EPOCHS"
  --lr "$LR"
  --feature-mode "$FEATURE_MODE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --max-rows "$MAX_TRAIN_ROWS"
  --val-max-rows "$VAL_MAX_ROWS"
  --log-interval "$LOG_INTERVAL"
  --prefetch-factor "$PREFETCH_FACTOR"
  --val-every "$VAL_EVERY"
  --optimizer "$OPTIMIZER"
  --weight-decay "$WEIGHT_DECAY"
  --scheduler "$SCHEDULER"
  --warmup-ratio "$WARMUP_RATIO"
  --min-lr-ratio "$MIN_LR_RATIO"
  --loss "$LOSS_FN"
  --label-smoothing "$LABEL_SMOOTHING"
  --wdl-temperature "$WDL_TEMPERATURE"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --use-ema "$USE_EMA"
  --ema-decay "$EMA_DECAY"
  --amp-dtype "$AMP_DTYPE"
  --cpu-threads "$CPU_THREADS"
  --cpu-interop-threads "$CPU_INTEROP_THREADS"
  --save-every "$SAVE_EVERY"
  --checkpoint-dir "$CKPT_DIR"
  --out "$OUT_NET"
)
if [[ ${#RESUME_ARG[@]} -gt 0 ]]; then
  train_cmd+=( "${RESUME_ARG[@]}" )
fi
"${train_cmd[@]}" | tee -a "$LOG"

if [[ "$OUT_NET" != "$ROOT/engine/cortex.nnue" ]]; then
  cp "$OUT_NET" "$ROOT/engine/cortex.nnue"
fi

if [[ "$AUTO_POST_VERIFY" == "1" ]]; then
  stage "4/5 post verify"
  "$ROOT/scripts/verify.sh" | tee -a "$LOG"
fi

if [[ "$AUTO_POST_EVAL" == "1" ]]; then
  if ! command -v cutechess-cli >/dev/null 2>&1; then
    echo "warning: missing dependency cutechess-cli; skipping AUTO_POST_EVAL" | tee -a "$LOG"
    AUTO_POST_EVAL=0
  fi
  if [[ "$AUTO_POST_EVAL" == "1" ]] && ! command -v bayeselo >/dev/null 2>&1; then
    echo "warning: missing dependency bayeselo; skipping AUTO_POST_EVAL" | tee -a "$LOG"
    AUTO_POST_EVAL=0
  fi
fi

if [[ "$AUTO_POST_EVAL" == "1" ]]; then
  stage "5/5 post SPRT/Elo evaluation"
  mkdir -p "$(dirname "$ELO_REPORT")"
  {
    echo "=== post-eval start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "candidate_net=$ROOT/engine/cortex.nnue"
    echo "match_out=$MATCH_OUT"
    echo "games=$EVAL_GAMES tc=$EVAL_TC threads=$EVAL_THREADS"
    echo "sprt alpha=$SPRT_ALPHA beta=$SPRT_BETA elo0=$SPRT_ELO0 elo1=$SPRT_ELO1"
  } | tee "$ELO_REPORT"

  GAMES="$EVAL_GAMES" \
  TC="$EVAL_TC" \
  THREADS="$EVAL_THREADS" \
  OPENINGS="$EVAL_OPENINGS" \
  MATCH_OUT="$MATCH_OUT" \
  ALPHA="$SPRT_ALPHA" \
  BETA="$SPRT_BETA" \
  ELO0="$SPRT_ELO0" \
  ELO1="$SPRT_ELO1" \
  CAND_NET="$ROOT/engine/cortex.nnue" \
    "$ROOT/scripts/sprt.sh" | tee -a "$LOG" | tee -a "$ELO_REPORT"

  {
    echo
    echo "=== post-eval end $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  } | tee -a "$ELO_REPORT"
fi

echo "== train_30h: done $(date -u) ==" | tee -a "$LOG"
