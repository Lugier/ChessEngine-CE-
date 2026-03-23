# Cortex Chess Engine

Constraint-first UCI chess engine built for practical strength on limited hardware.

Cortex combines a modern alpha-beta search stack with optional NNUE evaluation and a reproducible data/training pipeline. The repository is designed for Apple Silicon development (M2, 8 GB RAM) and optional cloud training windows (for example, 30h on RTX 3090).

## Executive Summary

- **What it is:** A production-style UCI engine with legal move generation, transposition table, search heuristics, and optional neural evaluation.
- **Why it exists:** Maximize Elo per resource unit under tight memory and compute constraints.
- **Current state:** End-to-end functional locally (build, verify, train, load net, search).
- **What it is not:** A claim of world #1 strength without formal match evidence (SPRT/Elo).

## Key Capabilities

| Area | Implementation |
|---|---|
| Core engine | Bitboards, legal move generation, FEN, Zobrist hashing, UCI protocol |
| Search | PVS, transposition table, null-move pruning, LMR, quiescence, SEE ordering, killer/history, aspiration |
| Rules in search | 50-move rule, repetition handling, simplified insufficient-material detection |
| Evaluation | Classical handcrafted evaluation + optional NNUE blend |
| NNUE formats | `CXN1` (legacy 768 features), `CXN2` (kingbucket / halfkp-like feature space) |
| SIMD | ARM64 NEON dot-product path with portable fallback |
| Data pipeline | Text/binpack conversion + out-of-core Lichess parquet quiet filtering |
| Training | PyTorch trainer with checkpoints/resume, val support, `CXN1`/`CXN2` export |
| Operations | RunPod scripts for checkpointed training and resume; outputs written to `runpod/out` and optionally copied to `engine/cortex.nnue` |
| Verification | One-command repo gate: build, perft, UCI smoke, dataprep, trainer smoke |

## Architecture at a Glance

1. **UCI input** enters the engine.
2. **Search** explores candidate lines using alpha-beta/PVS and pruning heuristics.
3. **Evaluation** scores positions using classical eval and optional NNUE blend.
4. **Best move** is returned as UCI output.
5. **Training loop** (offline): dataset -> binpack -> train -> export `cortex.nnue` -> reload in engine.

Detailed technical mapping: [`docs/ENGINE.md`](docs/ENGINE.md)

## Verified Baseline

Run this before commits, PRs, or cloud spend:

```bash
./scripts/verify.sh
```

What it validates:

- native build
- perft 1..5 from start position
- UCI handshake and command handling
- dataprep script behavior
- Python syntax checks
- search smoke test (classical mode)
- optional trainer smoke test + net export + NNUE search smoke

Fast mode without trainer:

```bash
SKIP_TRAINER=1 ./scripts/verify.sh
```

## Local Quickstart

```bash
./scripts/build.sh
./engine/cortex perft 5
printf 'uci\nisready\nposition startpos\ngo depth 6\nquit\n' | ./engine/cortex
```

## Data and Training

### Minimal smoke training (local)

```bash
python3 -m venv trainer/.venv
trainer/.venv/bin/pip install -r trainer/requirements.txt
python3 data/prepare_binpack.py data/sample_quiet.txt data/processed/sample.binpack
trainer/.venv/bin/python trainer/train_nnue.py \
  --data data/processed/sample.binpack \
  --out engine/cortex.nnue
```

### Large-scale out-of-core prep (Lichess parquet)

```bash
python3 -m venv data/.venv
data/.venv/bin/pip install -r data/requirements.txt
data/.venv/bin/python data/prepare_lichess_quiet.py \
  --parquet /path/to/lichess-evals.parquet \
  --train-out data/processed/train.binpack \
  --val-out data/processed/val.binpack \
  --summary-out data/processed/summary.json
```

## RunPod Training Flow (epoch-based)

The RunPod script is checkpointed and epoch-driven (`EPOCHS`, default `120`), not hard time-capped to exactly 30 hours.
If `train.binpack` / `val.binpack` are missing, it now automatically:

1. downloads large Lichess parquet shards on RunPod,
2. builds quiet `train.binpack` / `val.binpack`,
3. starts training.

```bash
# inside RunPod:
./runpod/train_30h.sh

# resume from latest checkpoint if interrupted:
./runpod/resume_30h.sh
```

Useful env overrides:

```bash
# use existing parquet shards (skip remote download if present)
PARQUET_GLOB='/workspace/Chess/data/raw/lichess-evals/*.parquet' ./runpod/train_30h.sh

# quick debug on subset only
MAX_ROWS=500000 ./runpod/train_30h.sh

# disable auto dataprep/download (expect prebuilt binpacks)
AUTO_PREP=0 ./runpod/train_30h.sh

# explicit training knobs
FEATURE_MODE=kingbucket LR=5e-4 EPOCHS=160 SAVE_EVERY=5 ./runpod/train_30h.sh

# if your runtime Python blocks pip installs, keep this at 0 (default) and use
# the Docker image dependencies; set 1 only as fallback
INSTALL_RUNTIME_DEPS=0 ./runpod/train_30h.sh
```

New trainer-quality and throughput knobs in `train_30h.sh`:

```bash
OPTIMIZER=adamw \
SCHEDULER=cosine \
LOSS_FN=kldiv \
USE_EMA=1 \
LABEL_SMOOTHING=0.01 \
WDL_TEMPERATURE=1.2 \
GRAD_CLIP_NORM=1.0 \
AMP_DTYPE=fp16 \
CPU_THREADS=4 CPU_INTEROP_THREADS=1 \
./runpod/train_30h.sh
```

After training:

```bash
LARGE_NET_FILE=engine/cortex.nnue ./scripts/verify.sh
```

## Strength Evaluation Workflow

By default, match scripts compare a classical baseline (`UseNNUE=false`) against an NNUE candidate. Set `BASE_NET` and `CAND_NET` explicitly for net-vs-net tests:

```bash
./scripts/sprt.sh
```

Note: if `bayeselo` is not installed, `scripts/sprt.sh` still runs matches and writes PGN output, but does not produce an automated accept/reject verdict.

For practical use:

- ensure `cutechess-cli` is installed
- provide explicit net paths for baseline/candidate as needed
- run sufficient games for stable decisions

## 4h Selfplay + Finetune Flow

For faster iteration cycles on one RTX 3090 + high CPU concurrency:

```bash
./runpod/train_4h_selfplay.sh
```

Pipeline stages:

1. selfplay generation via `cutechess-cli` with high `-concurrency`
2. PGN conversion to binpack via `data/pgn_to_binpack.py`
3. finetune with mixed supervised + selfplay data (`--aux-data`, `--aux-ratio`)
4. quick SPRT gate

Useful overrides:

```bash
CUTECHESS_BIN=/workspace/tools/cutechess-build/cutechess-cli \
OUT_DIR=/workspace/Chess/runpod/out-4h \
SELFPLAY_CONCURRENCY=48 \
SELFPLAY_GAMES=1200 \
SELFPLAY_RATIO=0.25 \
FEATURE_MODE=kingbucket \
BATCH_SIZE=6144 NUM_WORKERS=28 \
EPOCHS=48 \
./runpod/train_4h_selfplay.sh
```

Note: `scripts/cutechess_match.sh` now supports `CONCURRENCY`, `BASE_THREADS`,
`CAND_THREADS`, `BASE_HASH`, and `CAND_HASH` for scalable selfplay/eval runs.

## Current Constraints and Realistic Positioning

This project is engineered for high efficiency under constrained hardware, not for unsupported marketing claims.

- Stronger than sample baseline: realistic with quality data and measured tuning.
- Better than top global engines: not a valid claim without large-scale training infrastructure and formal benchmarks.

## Repository Structure

| Path | Purpose |
|---|---|
| `engine/` | C++ engine code |
| `data/` | dataprep scripts (`prepare_binpack.py`, `prepare_lichess_quiet.py`) |
| `trainer/` | PyTorch training and NNUE export |
| `runpod/` | cloud training scripts and container setup |
| `scripts/` | verify, benchmark, match, SPRT helpers |
| `docs/` | technical design and Gemini alignment |
| `Gemini.md` | strategy brief and constraints |

## Roadmap

- Expand and harden large-scale dataset quality controls
- Complete full halfkp-style incremental path for non-legacy mode
- Improve match automation and statistical gating for promotion decisions
- Continue CPU-side optimization based on measured profiling data

## License

No explicit project license is currently declared. Add a license before redistribution.
