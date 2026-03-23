#!/usr/bin/env python3
"""
Binpack writer for trainer/train_nnue.py.

Text formats (pipe-separated, UTF-8):
  fen | w | d | l     — explicit WDL (should sum ~1)
  fen | cp            — centipawns from White POV; converted via cp_to_wdl()

Binary layout: uint32 count; then per record uint16 fen_len, fen bytes, 3×float32 WDL.
Little-endian. No I/O besides read argv[1], write argv[2].

Large Lichess/SF-eval pipelines stay external (out-of-core); this script is the
stable ingestion boundary into training.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def cp_to_wdl(cp: float, scale: float = 400.0) -> tuple[float, float, float]:
    """
    Gemini.md (§5.2): Centipawns → glatter WDL-Raum statt roher MSE auf cp.
    Zwei Logits Win/Loss um 0, Remis zentral — Softmax, Summe = 1.
    """
    z = cp / scale
    a, b, c = z, 0.0, -z
    m = max(a, b, c)
    ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
    s = ea + eb + ec
    return ea / s, eb / s, ec / s


def parse_line(line: str, cp_scale: float) -> tuple[str, tuple[float, float, float]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) == 4:
        fen, w, d, l_ = parts
        return fen, (float(w), float(d), float(l_))
    if len(parts) == 2:
        fen, cp = parts
        cpv = max(-2000.0, min(2000.0, float(cp)))
        w, d, l_ = cp_to_wdl(cpv, scale=cp_scale)
        return fen, (w, d, l_)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", type=Path)
    ap.add_argument("output_bin", type=Path)
    ap.add_argument(
        "--cp-scale",
        type=float,
        default=400.0,
        help="cp→WDL Steilheit (Gemini: geglättete Sigmoid/Softmax-Kette)",
    )
    args = ap.parse_args()
    if args.cp_scale <= 0:
        raise SystemExit("--cp-scale must be > 0")

    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    with args.output_bin.open("wb") as f:
        f.write(struct.pack("<I", 0))
        n = 0
        with args.input_txt.open("r", encoding="utf-8") as inp:
            for line in inp:
                p = parse_line(line, cp_scale=args.cp_scale)
                if not p:
                    continue
                fen, (w, d, l_) = p
                # sanitize explicit WDL input
                if not (math.isfinite(w) and math.isfinite(d) and math.isfinite(l_)):
                    continue
                s = w + d + l_
                if s <= 1e-9:
                    continue
                w, d, l_ = w / s, d / s, l_ / s
                b = fen.encode("utf-8")
                if len(b) == 0 or len(b) > 0xFFFF:
                    continue
                f.write(struct.pack("<H", len(b)))
                f.write(b)
                f.write(struct.pack("<fff", w, d, l_))
                n += 1
        f.seek(0)
        f.write(struct.pack("<I", n))
    print(f"wrote {n} positions -> {args.output_bin}")


if __name__ == "__main__":
    main()
