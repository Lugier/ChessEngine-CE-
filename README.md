# Cortex — constraint-first chess engine (Alpha–Beta + NNUE)

Local-first project layout aligned with the autonomous build plan: C++ UCI engine, classic eval fallback, compact NNUE (768→256→scalar export), Python dataprep and PyTorch trainer for Apple Silicon (MPS) or CPU.

## Abgleich mit `Gemini.md`

Die Strategie aus **Gemini.md** ist die Referenz für Architektur und Ressourcen. Stand dieses Repos:

| Gemini-Vorgabe | Umsetzung / Status |
|----------------|-------------------|
| **Constraint-first**, kein AlphaZero/MCTS/Lc0, keine Searchless-Transformer, kein LLM/MLX als Engine | Bewusst nicht implementiert; README und Code bleiben bei Alpha–Beta + kleinem NNUE. |
| **Hybrid: C++ Alpha–Beta + NNUE**, taktisch Suche, positionell Netz | Ja: PVS, TT, Nullmove, LMR, Quiescence + optional `cortex.nnue` und klassische Eval als Mischung. |
| **Training: Distillation** aus fremd evaluierten Daten, **kein** Self-Play-RL | Trainer ist Supervised Learning auf (FEN → WDL); kein MCTS/RL. |
| **WDL statt roher cp-MSE** (§5.2) | `data/prepare_binpack.py`: Centipawns → **Softmax-WDL** über Logits \((cp/s, 0, -cp/s)\), `--cp-scale` steuerbar. |
| **Quiet positions** filtern (§5.2) | Out-of-Core-Pipeline (DuckDB/Polars) für große Lichess-Exports ist **noch** anzubinden; `sample_quiet.txt` nur Rauchtest. |
| **Binärformat statt Massen-FEN auf GPU** (§5.2) | Einfaches `.binpack` (Count + FEN + 3×float WDL); kompatibel mit lokalem Trainer. |
| **M2: Clang-Tuning, RAM für TT** (§2.1, §6.1) | `scripts/build.sh` und CMake nutzen auf **Darwin arm64** `-mcpu=apple-m2`. TT über UCI `Hash` (MB); bei **8 GB RAM** empfohlen: erst messen, typisch **512–2048 MB**, nicht „maximal blind“ um Swap zu vermeiden. |
| **NNUE: Quantisierung, SIMD/NEON, ClippedReLU** (§4.2) | Inferenz **int16**; Hidden-Layer **ClippedReLU** (Clip 32767). **NEON dot-products / inkrementelles HalfKP** = nächster Schritt (aktuell Full-Refresh 768-Plane). |
| **Quiescence gegen Quiet-Trainings-Blindheit** (§8) | Quiescence sucht Schlagfolgen; deckt Gemini-Risiko „nur ruhige Daten“ teilweise ab. |
| **RTX 3090 / RunPod ≤ 30 h** (§2.2, §6.2) | Nicht angebunden; wenn aktiv: `nnue-pytorch` (Stockfish) oder diesen Trainer mit großem Binpack + großer Batch-Size evaluieren. |
| **Phasenplan** (§7) | Phase 0–2 teilweise da; Phase 1 Out-of-Core + echtes Quiet-Labeling offen; Phase 3–4 Cloud/Tuning offen. |

Details und Formeln liegen im vollen Text von `Gemini.md` im Repo-Root.

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
