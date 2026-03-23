# Cortex — technische Referenz

Dieses Dokument beschreibt **was** der Code tut, **warum** die Entscheidungen so getroffen wurden, und **wie** Korrektheit geprüft wird. Es ersetzt lange Kommentarblöcke in der Implementierung zugunsten **leaner** Quellen.

## Zielbild und Grenzen

- **Ziel (Gemini.md):** Hybrid aus **CPU-Alpha–Beta** und **kleinem NNUE**, Distillation aus gelabelten Daten, Arbeitsspeicher und Trainingsbudget im Rahmen (z. B. 8 GB, begrenzte GPU-Zeit).
- **Nicht-Ziel:** „Stärkste Engine am Markt“ oder **Stockfish / Lc0 schlagen**. Das verlangt andere Architektur (z. B. HalfKP, inkrementelles NNUE, SIMD-tuned Inferenz), **Größenordnungen mehr Daten** und **SPRT-gemessene** Spielstärke. Cortex ist eine **korrekte, erweiterbare Basis** unter bewussten Constraints.
- **Stärke steigern (realistisch):** mehr/bessere Trainingsdaten (ruhige Stellungen, SF-Eval), längeres Training, größere TT (`Hash`), Texel-/Tuning, dann **Matches messen** (Cutechess + SPRT).

## Verifizierbare Korrektheit

| Prüfung | Bedeutung |
|--------|-----------|
| **Perft 1–5** (Startstellung) | Zuggenerierung inkl. Rochade, EP, Promotion; Zähler **4865609** bei Tiefe 5. |
| **UCI** | Protokoll-Handshake, `position`, `go`; Suche liefert `bestmove`. |
| **`scripts/verify.sh`** | Baut die Engine, führt Perft + UCI + Dataprep + (optional) Trainer aus; **vor jedem Merge** laufen lassen. |

Was **nicht** durch Perft abgedeckt ist: **Suchheuristiken** (LMR, Nullmove, Aspiration) können theoretisch **Raritäts-Bugs** haben; dagegen helfen **Spieltests** und feste Teststellungen, nicht nur Perft.

## Schichtenmodell

1. **Repräsentation:** Bitboards + `Board` (FEN, Zobrist-Hash, legaler Zug, Remis-Helfer).
2. **Suche:** Iterative Vertiefung an der Wurzel, **PVS** im Baum, **Transposition Table**, **Nullmove**, **LMR** auf spätere Züge, **Quiescence** nur mit **lauten** Zügen (Schlag/EP/Promotion).
3. **Ordnung:** TT-Zug zuerst, dann **MVV-LVA**-artige Schätzung für Schläge, **SEE** in der Quiescence zur Sortierung, **Killer** und **History** für leise Züge; ab Tiefe 4 **Aspiration** an der Wurzel mit Neu-Suche bei Fail-Low/High.
4. **Eval:** Klassisch (Material + PST, **König tapering** nach Spielphase, **Läuferpaar** stärker im Endspiel). Optional **NNUE** (`cortex.nnue`): bei aktivem Netz **Mischung** ` (3 * nnue + 1 * classic) / 4 ` — stabilisiert kleine/schwache Netze und verhindert komplettes Wegdriften von taktisch plausibler klassischer Eval.

## Wichtige Implementierungsdetails

### Wiederholung und Remis in der Suche

- **`Board::game_rep_keys`:** Hashs entlang der **tatsächlichen Partiezüge** (UCI); wird beim Start der Suche in `rep_stack` kopiert; **nicht** durch internes Do/Undo verändert.
- **Suchpfad:** Nach jedem `do_move` wird der neue Positions-Hash auf `rep_stack` gelegt; **Dreifachwiederholung** zählt Vorkommen von `b.hash` auf dem Stack.
- **50-Züge:** `halfmove >= 100` → Remis in der Suche.
- **Materialremis:** vereinfacht (z. B. KK, K vs KN/KB); **kein** vollständiger FIDE-Katalog (z. B. gleichfarbige Läufer KB vs KB).

### NNUE-Format `CXN1`

- Siehe `engine/src/nnue.cpp`: Magic `CXN1`, Version, `out_scale`, Gewichte **int16**, **ClippedReLU** in der versteckten Schicht (Gemini.md §4.2).
- **Features:** 768 = 64 Felder × 12 Stücktypen (pro Feld höchstens ein „1“-Eintrag), **Full-Refresh** pro Eval — bewusst einfach; **kein** Stockfish-HalfKP.

### NNUE-Format `CXN2` (halfkp-like) + inkrementeller Pfad

- `trainer/train_nnue.py --feature-mode kingbucket` exportiert `CXN2` (Version 2).
- `engine/src/nnue.cpp` erkennt `CXN1` und `CXN2`. `CXN2` nutzt kingbucket-Features (16 Buckets × 64 × 10).
- Inkrementelles Hidden-Update ist derzeit für `CXN1` aktiv; `CXN2` fällt aktuell auf Recompute zurück.
- SIMD-Beschleunigung läuft über `engine/src/nnue_neon.cpp` mit portablem Fallback.

### Trainer ↔ Engine

- `data/prepare_binpack.py`: Textzeilen → Binärpack (Anzahl, pro Zeile FEN-Länge, FEN, drei **float** WDL). Centipawns optional → WDL per Softmax (`cp_to_wdl`).
- `trainer/train_nnue.py`: Netz `768 → ReLU → 256 → **3**` (WDL-Logits), Training mit MSE auf WDL; **Export** faltet die zweite Schicht auf **einen** skalaren Ausgang (`w2[0]`, `b2[0]`), damit die Engine-Einheit (`w2` Länge 256) konsistent bleibt. Dokumentiert im Quellkopf.

## Quellmodul-Index (Navigation)

Ausführliche Logik steht bewusst **hier**; die `.cpp`-Dateien tragen nur **kurze** Köpfe, damit der Code lesbar und **lean** bleibt.

| Datei | Aufgabe |
|--------|--------|
| `engine/src/main.cpp` | Einstieg: `perft <n>` oder UCI-Schleife |
| `engine/src/uci.cpp` | UCI-Protokoll, Optionen (`Hash`, `UseNNUE`), `go`/`stop` |
| `engine/src/board.cpp` | FEN, Züge, Hash-Update, `game_rep_keys` |
| `engine/src/movegen.cpp` | Pseudozüge + Filter auf legal (König nicht im Schach) |
| `engine/src/bitboard.cpp` | Attack-Tabellen, Linien/Zwischenfelder |
| `engine/src/zobrist.cpp` | Deterministische Keys (fixer Seed) |
| `engine/src/tt.cpp` | TT: Größe ~2^n Buckets, probe/store |
| `engine/src/see.cpp` | SEE für Schlag-Sortierung in Quiescence |
| `engine/src/search.cpp` | PVS, Nullmove, LMR, Aspiration, Killer/History |
| `engine/src/eval_classic.cpp` | Klassische Eval + König-Taper + Läuferpaar |
| `engine/src/nnue.cpp` | `CXN1`/`CXN2` laden, Inferenz, Cache-/Inkrement-Logik |
| `engine/src/nnue_neon.cpp` | Dot-Product ARM64/NEON + Fallback |
| `scripts/verify.sh` | Automatischer Korrektheits- und Smoke-Gate |
| `scripts/build.sh` | `clang++` Release-Binary `engine/cortex` |
| `data/prepare_binpack.py` | Text → Binpack für Training |
| `data/prepare_lichess_quiet.py` | Out-of-core Lichess parquet → quiet train/val binpack |
| `trainer/train_nnue.py` | WDL-Training, Export `cortex.nnue` |
| `runpod/train_30h.sh` | Strikter 30h-Trainingslauf mit Checkpoints |
| `runpod/resume_30h.sh` | Resume vom letzten Checkpoint |
| `scripts/sprt.sh` | Akzeptanz-Gate (Match + Auswertung) |

## Änderungen an der Suche / Eval

- Vor größeren Eingriffen: **`./scripts/verify.sh`** (oder `SKIP_TRAINER=1` für Schnelllauf).
- Neue Heuristiken: dokumentieren **hier** oder in **kurzen** File-Headern (1 Absatz), nicht mit Zeilenkommentaren pro Statement.

## Abgleich `Gemini.md` (Spezifikation → dieses Repo)

`Gemini.md` ist der **strategische Aufschlag** (Constraint-First, Hybrid AB+NNUE, Distillation, M2 + optional 3090). Er ist als **Markdown-Kapitel** im Repo-Root gepflegt; **operativ** gelten README und dieses Dokument für den Code-Abgleich.

| Gemini-Thema | § (ca.) | Status im Repo |
|----------------|---------|----------------|
| Kein MCTS/AlphaZero, kein Searchless-Transformer | 3 | Nicht implementiert (bewusst). |
| Hybrid Alpha–Beta + NNUE, PVS/NMP/LMR/Quiescence | 4.2–4.3 | Implementiert (`search.cpp`); NNUE mit `CXN1`/`CXN2`, inkrementell aktuell für `CXN1`, kein vollständiges Stockfish-HalfKP. |
| WDL statt roher cp-MSE, Binpack | 5.2 | `prepare_binpack.py`, Trainer mit WDL-Head. |
| Quiet-Filter, Out-of-Core (DuckDB), große Lichess-Sets | 5.2, 6.1 | Grundpfad umgesetzt (`data/prepare_lichess_quiet.py`), produktive Datenbereitstellung bleibt operativ. |
| int16/ClippedReLU, SIMD/NEON | 4.2 | int16 + ClippedReLU; NEON-Dot-Product mit portablem Fallback implementiert. |
| TT groß aber ohne Swap | 2.1, 6.1 | UCI `Hash` **1…3072 MB** mit Clamp; Mattwerte **ply-normalisiert** (`to_tt`/`from_tt`); RAM bewusst wählen (Swap vermeiden). |
| 3090 / nnue-pytorch / RunPod | 6.2, Phase 3 | `runpod/`-Skripte vorhanden (train/resume/Dockerfile); optionaler Wechsel zu `nnue-pytorch` weiterhin offen. |
| „Beste Engine am Markt“ / Welt-Elo | divers | **Kein Projektziel**; Stärke nur durch Daten + Messung (SPRT). |

## Glossar (kurz)

| Begriff | Rolle in Cortex |
|--------|------------------|
| **PVS** | Principal Variation Search; Null-Fenster für Nachzüge, Re-Search bei Treffer. |
| **SEE** | Static Exchange Evaluation; sortiert Schlagfolgen in Quiescence. |
| **LMR** | Late Move Reductions; reduziert Tiefe für späte, ruhige Züge. |
