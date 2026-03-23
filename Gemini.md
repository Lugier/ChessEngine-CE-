# Systemdesign: Schach-Engine unter extremen Hardware-Restriktionen

> **Navigation:** Dieses Dokument ist der **strategische Auftrag** (Constraint-First, Hybrid-Suche + NNUE, Distillation). **Was im Repo wie umgesetzt ist:** [`docs/ENGINE.md`](docs/ENGINE.md) und [`README.md`](README.md).

---

## 1. Architektonische Grundlagen

Computerschach auf Weltklasse-Niveau kostet typischerweise massive Rechenressourcen (z. B. AlphaZero-Self-Play). Dieses Projekt wählt das Gegenteil: **asymmetrische, harte Limits** — etwa **Apple M2 mit 8 GB UMA** (zeitlich unbegrenzt nutzbar für Dev, Build, Dataprep) und **optional bis zu ~30 Stunden NVIDIA RTX 3090** (RunPod) ausschließlich für **konzentriertes Supervised Training**, nicht für Datenexploration oder Self-Play in großem Stil.

**Leitmotiv: Constraint-First.** Entscheidungen werden an **„Elo pro Ressourceneinheit“** gemessen. Ansätze, die primär über extreme Skalierung gewinnen (**MCTS + tiefes CNN**, **Searchless-Transformer**, **LLM als Zugmotor**), sind hier **ausgeschlossen**.

**Kernempfehlung:** **Hybrid** aus **hochoptimierter Alpha–Beta-Suche in C++** und **kleinem, quantisiertem NNUE** für die Eval. Taktik bleibt in der Suche; Positionelles steckt im Netz. **Wissens-Destillation** aus fremd (z. B. Stockfish) gelabelten Daten ersetzt RL-Self-Play. Das M2 übernimmt Preprocessing, Kompilierung und Engine-Lauf; ARM64-Tuning (`-mcpu=apple-m2`) maximiert NPS im Rahmen der Hardware.

## 2. Physikalische und ökonomische Constraints

### 2.1 MacBook M2 (8 GB UMA)

UMA teilt RAM zwischen CPU, GPU und NPU. **8 GB Gesamt** minus System (~2,5–3,5 GB) lässt für Engine + Netz + OS-Rest oft nur **~4,5–5 GB** „Nutzkopf“. **Swapping auf SSD** zerstört **NPS** — daher: kleiner **Binary-/Netz-Footprint**, große **Transpositionstabelle** nur innerhalb des verbleibenden RAM ohne Swap. **Keine** schweren Modelle (LLM, große ResNets), die TT und Suche verdrängen.

### 2.2 RTX 3090 (Zeitbudget ~30 h)

Die GPU ist **knapp und teuer**. **Kein** Self-Play, kein massives Hyperparameter-Raten, kein Training riesiger Transformer in diesem Fenster. Die GPU ist **Injektionskanal**: vorgefiltertes, binär verpacktes Wissen → **flaches NNUE** in Stunden, nicht Tage sinnloser Pipeline-Fehler.

### 2.3 Metrik: Elo pro Ressourceneinheit

| Architektur              | Inferenz (RAM) | Trainingsbedarf   | Erwartung unter knappem GPU-Budget |
|--------------------------|----------------|-------------------|-------------------------------------|
| AlphaZero (MCTS+CNN)     | hoch           | extrem hoch       | schlechte Ausbeute                  |
| Searchless Transformer   | sehr hoch      | hoch              | schlechte Ausbeute                  |
| Nur Klassik              | minimal        | keine GPU         | moderat                             |
| **Alpha–Beta + NNUE**    | **sehr gering**| **moderat**       | **bestes Kosten-Nutzen im Limit**   |

## 3. Verworfene Paradigmen (bewusst)

### 3.1 RL / MCTS (z. B. Lc0-Stil)

MCTS skaliert mit **Eval-Durchsatz**. Auf dem M2 bricht NPS ein; CPU↔GPU-Overhead für jede Stellung ist Gift. **Nicht Ziel dieses Repos.**

### 3.2 Searchless Chess / große Transformer

Starke Forschung, aber **hunderte MB–GB** Parameter + KV-Cache; **keine** integrierte Suchhistorie für **Dreifach** etc. **Nicht vereinbar mit 8 GB + kleiner Engine.**

### 3.3 MLX / LLM für Züge

Tokenisierung von FEN und autoregressive Vorhersage sind **Größenordnungen langsamer** als klassische Engine. **Ungeeignet.**

## 4. Architektonischer Kern: NNUE + Alpha–Beta

### 4.1 Idee des NNUE

Extrem flach, **sparse** Inputs (viele Features, wenige aktiv — z. B. HalfKP in der Literatur). **Inkrementelle Updates** (Akkumulator ± wenige Zeilen) sind der Standard in Top-Engines; **dieses Repo** nutzt derzeit ein **einfacheres 768-Plane Full-Refresh** als Übergang (siehe `docs/ENGINE.md`).

### 4.2 Quantisierung & M2

**int16/int8-Inferenz**, **ClippedReLU**, Skalierung mit **Zweierpotenzen** (Bit-Shifts). **NEON/SIMD** ist vorgesehen für spätere Optimierung; Pflicht ist **korrekte** Funktion zuerst.

### 4.3 Suchalgorithmus

Klassisches **Alpha–Beta** mit **PVS**, **Nullmove**, **LMR**, **Quiescence** — genau die Heuristiken, die **taktische Lücken** einer statischen Eval (und von „nur ruhigen“ Trainingsdaten) abfangen.

## 5. Datenstrategie: Destillation

### 5.1 Quellen

Öffentliche **Lichess-/SF-evaluierte** Datensätze (Hugging Face, gefilterte Quiet-Sets, etc.) — **nicht** selbst generieren im GPU-Fenster.

### 5.2 Selektion & Zielgröße

**Centipawn-MSE** auf Extremwerten ist instabil → Ziel ist **WDL** (Win/Draw/Loss) im \[0,1\]. **Quiet positions** filtern (keine hängenden Schläge/Schach/Promo in der Stellung), damit das Netz nicht „Taktik raten“ muss, die nur die Suche lösen soll. **Binpack** statt Massen-FEN auf der GPU für I/O.

## 6. Ressourcen-Split

### 6.1 M2 (unbegrenzte Zeit)

- **Out-of-Core** (DuckDB/Polars): große Parquet/CSV **stückweise** → Filter → **Binpack** auf SSD.  
- **C++-Engine**, **Clang**, `-mcpu=apple-m2` wo möglich.  
- **TT-Größe** bewusst wählen; kein unkontrolliertes Wachstum (harte UCI-Grenzen).

### 6.2 RTX 3090 (~30 h)

Nur wenn **Binpacks validiert** sind. Optional **nnue-pytorch** (Stockfish-Ökosystem) oder der **kleine PyTorch-Trainer** im Repo — siehe `trainer/`. Große **Batch-Größen**, **Mixed Precision**, **Early Stopping** / Validierung gegen Overfitting.

## 7. Phasenplan (Referenz)

| Phase | Ort      | Inhalt |
|-------|----------|--------|
| 1     | M2       | Download/Filter/Quiet → WDL → **train/val.binpack** |
| 2     | M2       | C++ Core: Bitboards, Movegen, **PVS/NMP/LMR/Qsearch**, NNUE-Pfad |
| 3     | 3090     | Training, Export quantisiertes Netz |
| 4     | M2       | Integration, Tuning, **gemessene** Matches (SPRT) |

## 8. Risiken & Mitigation

- **Overfitting:** Validierungs-Holdout, Regularisierung, Early Stopping.  
- **Quiet-Only-Blindheit:** **Quiescence** und starke Suche im Code.  
- **OOM/Swap:** feste TT-Allokation, keine Allokation in der innersten Suchschleife.  
- **Cloud-Zeitverlust:** Skripte **lokal** bis `verify` grün; Docker/ reproduzierbare Images für GPU.

## 9. Ausblick

Ziel ist **maximale Stärke unter den genannten Grenzen**, nicht absolutes Weltniveau ohne Messung. **Nächste Schritte:** große **ruhige** Datenpipelines anbinden, Netz/Architektur skalieren, **Elo durch SPRT** belegen.

---

*Hinweis: Eine ältere, einzeilige Exportfassung desselben Plans kann in Backups vorkommen; diese Datei ist die **wartbare** Markdown-Version für Menschen und Agenten.*
