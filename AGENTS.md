# Hinweise für automatische Agenten

1. **Vor Änderungen am Engine- oder Python-Code:** `./scripts/verify.sh` ausführen. Schnell ohne Torch: `SKIP_TRAINER=1 ./scripts/verify.sh`. Bei Exit-Code ≠ 0 nicht committen.
2. **Architektur-Ziel** steht in `Gemini.md` (Hybrid Alpha–Beta + NNUE, Distillation, 8 GB / 30 h GPU als Rahmen).
3. **Build:** `scripts/build.sh` bevorzugen; CMake ist optional (`engine/CMakeLists.txt`).
