# Cortex — constraint-first chess engine (Alpha–Beta + NNUE)

Local-first project layout aligned with the autonomous build plan: C++ UCI engine, classic eval fallback, compact NNUE (768→256→scalar export), Python dataprep and PyTorch trainer for Apple Silicon (MPS) or CPU.

## Build (macOS, no CMake required)

```bash
./scripts/build.sh
```

Produces `engine/cortex`. With CMake installed, you can also use `engine/CMakeLists.txt`.

## Quick checks

```bash
./engine/cortex perft 5    # expect 4865609 from the initial position
printf 'uci\nisready\nposition startpos\ngo depth 6\nquit\n' | ./engine/cortex
```

## NNUE file (`cortex.nnue`)

Binary format `CXN1` (see `engine/src/nnue.cpp`): int16 weights for a full refresh evaluator (768 sparse piece–square features, 256 ReLU, scalar out). The engine blends NNUE with the hand-crafted eval when a net is loaded.

Train and export into `engine/` (uses a project venv):

```bash
python3 -m venv trainer/.venv
trainer/.venv/bin/pip install -r trainer/requirements.txt
python3 data/prepare_binpack.py data/sample_quiet.txt data/processed/sample.binpack
trainer/.venv/bin/python trainer/train_nnue.py --data data/processed/sample.binpack --out engine/cortex.nnue
```

The sample lines in `data/sample_quiet.txt` are only for **offline smoke tests**. For real strength, replace this with a filtered Lichess (or similar) Stockfish-eval set and map centipawns → WDL in `data/prepare_binpack.py`.

## Directories

| Path | Role |
|------|------|
| `engine/` | C++ engine: bitboards, legal movegen, PVS search, TT, null move, LMR, quiescence, UCI |
| `data/` | Dataset prep scripts; keep large binaries under `data/raw/` or `data/processed/` (gitignored) |
| `trainer/` | PyTorch training + `CXN1` export |
| `scripts/` | `build.sh` and future benchmarks |

## Known limitations (v0)

- NNUE architecture is a **deliberately small** dense 768-plane encoding for pipeline shakedown, not HalfKP/Stockfish-compatible training.
- No repetition / 50-move draw detection in search yet.
- RunPod / RTX training Dockerfile is intentionally omitted here; add it when you are ready to spend GPU budget.

## License

Add a license if you redistribute; default is all rights reserved until you choose one.
