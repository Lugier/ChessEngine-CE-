#!/usr/bin/env python3
"""
Convert lines of `fen | w | d | l` (or `fen | cp`) into a simple packed binary
for the trainer. No network I/O.

Example line (WDL sum ~1.0):
  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0.33 | 0.34 | 0.33
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

    rows: list[tuple[str, tuple[float, float, float]]] = []
    for line in args.input_txt.read_text().splitlines():
        p = parse_line(line, cp_scale=args.cp_scale)
        if p:
            rows.append(p)

    out = bytearray()
    out += struct.pack("<I", len(rows))
    for fen, (w, d, l_) in rows:
        b = fen.encode("utf-8")
        out += struct.pack("<H", len(b))
        out += b
        out += struct.pack("<fff", w, d, l_)

    args.output_bin.write_bytes(out)
    print(f"wrote {len(rows)} positions -> {args.output_bin}")


if __name__ == "__main__":
    main()
