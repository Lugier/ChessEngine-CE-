#!/usr/bin/env python3
"""
Out-of-core Lichess parquet -> quiet train/val binpack.

Input parquet is expected to contain at least: fen, cp, mate, depth.
The script streams rows through DuckDB, applies lightweight SQL filters first,
then validates "quiet" with python-chess before writing binary packs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

import chess
import duckdb


def cp_to_wdl(cp: float, scale: float = 400.0) -> tuple[float, float, float]:
    z = cp / scale
    a, b, c = z, 0.0, -z
    m = max(a, b, c)
    ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
    s = ea + eb + ec
    return ea / s, eb / s, ec / s


def quiet_position(fen: str) -> bool:
    try:
        b = chess.Board(fen)
    except ValueError:
        return False
    if b.is_check():
        return False
    # Keep only calm positions: no immediate capture/promotion candidate.
    for mv in b.legal_moves:
        if b.is_capture(mv):
            return False
        if mv.promotion is not None:
            return False
    return True


@dataclass
class Stats:
    scanned_rows: int = 0
    passed_sql: int = 0
    passed_quiet: int = 0
    train_rows: int = 0
    val_rows: int = 0
    skipped_bad_fen: int = 0
    skipped_non_quiet: int = 0


class BinpackWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("wb")
        self.count = 0
        self.f.write(struct.pack("<I", 0))  # patched on close

    def add(self, fen: str, wdl: tuple[float, float, float]) -> None:
        w, d, l_ = wdl
        b = fen.encode("utf-8")
        self.f.write(struct.pack("<H", len(b)))
        self.f.write(b)
        self.f.write(struct.pack("<fff", w, d, l_))
        self.count += 1

    def flush(self) -> None:
        self.f.seek(0)
        self.f.write(struct.pack("<I", self.count))
        self.f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--cp-scale", type=float, default=400.0)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-depth", type=int, default=16)
    ap.add_argument("--cp-cap", type=float, default=1200.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = Stats()
    train = BinpackWriter(args.train_out)
    val = BinpackWriter(args.val_out)

    con = duckdb.connect(database=":memory:")
    limit_sql = f"LIMIT {args.max_rows}" if args.max_rows > 0 else ""
    q = f"""
        SELECT fen, cp, mate, depth
        FROM read_parquet('{args.parquet.as_posix()}')
        WHERE depth >= {int(args.min_depth)}
          AND mate IS NULL
          AND cp IS NOT NULL
          AND abs(cp) <= {float(args.cp_cap)}
        {limit_sql}
    """
    rel = con.sql(q)

    offset = 0
    while True:
        chunk = rel.limit(args.chunk_size, offset).fetchall()
        if not chunk:
            break
        stats.passed_sql += len(chunk)
        offset += len(chunk)
        for fen, cp, _mate, _depth in chunk:
            stats.scanned_rows += 1
            if not isinstance(fen, str):
                stats.skipped_bad_fen += 1
                continue
            cpf = float(cp)
            if not math.isfinite(cpf):
                stats.skipped_bad_fen += 1
                continue
            if not quiet_position(fen):
                stats.skipped_non_quiet += 1
                continue
            stats.passed_quiet += 1
            wdl = cp_to_wdl(cpf, scale=args.cp_scale)
            if rng.random() < args.val_ratio:
                val.add(fen, wdl)
                stats.val_rows += 1
            else:
                train.add(fen, wdl)
                stats.train_rows += 1

    train.flush()
    val.flush()
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(asdict(stats), indent=2) + "\n")
    print(f"train={stats.train_rows} val={stats.val_rows} summary={args.summary_out}")


if __name__ == "__main__":
    main()
