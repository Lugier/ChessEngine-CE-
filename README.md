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
| Operations | RunPod scripts for 30h training flow, resume, and artifact handoff |
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

## RunPod 30h Standard Flow

```bash
# inside your cloud environment with data available:
./runpod/train_30h.sh

# resume from latest checkpoint if interrupted:
./runpod/resume_30h.sh
```

After training:

```bash
LARGE_NET_FILE=engine/cortex.nnue ./scripts/verify.sh
```

## Strength Evaluation Workflow

Use match scripts to compare baseline vs candidate nets:

```bash
./scripts/sprt.sh
```

For practical use:

- ensure `cutechess-cli` is installed
- provide explicit net paths for baseline/candidate as needed
- run sufficient games for stable decisions

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
