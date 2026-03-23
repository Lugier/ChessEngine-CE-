#!/usr/bin/env python3
"""
Convert lines of `fen | w | d | l` (or `fen | cp`) into a simple packed binary
for the trainer. No network I/O.

Example line (WDL sum ~1.0):
  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0.33 | 0.34 | 0.33
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def parse_line(line: str) -> tuple[str, tuple[float, float, float]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) == 4:
        fen, w, d, l_ = parts
        return fen, (float(w), float(d), float(l_))
    if len(parts) == 2:
        fen, cp = parts
        # Map centipawns to crude WDL (toy; replace with sigmoid in production).
        v = max(-1000, min(1000, float(cp))) / 1000.0
        w = 0.5 + v * 0.25
        d = 0.25
        l_ = 1.0 - w - d
        if l_ < 0:
            l_ = 0.0
            d = 1.0 - w
        return fen, (w, d, l_)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", type=Path)
    ap.add_argument("output_bin", type=Path)
    args = ap.parse_args()

    rows: list[tuple[str, tuple[float, float, float]]] = []
    for line in args.input_txt.read_text().splitlines():
        p = parse_line(line)
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
