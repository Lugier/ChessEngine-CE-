#!/usr/bin/env python3
"""
Out-of-core Lichess parquet -> quiet train/val binpack.

Input parquet is expected to contain at least: fen, cp, mate, depth.
The script streams rows through DuckDB, applies lightweight SQL filters first,
then validates "quiet" with python-chess before writing binary packs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import struct
import tempfile
import time
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
    skipped_bad_cp: int = 0
    skipped_too_long_fen: int = 0


class BinpackWriter:
    def __init__(self, path: Path):
        self.final_path = path
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.final_path.parent,
            prefix=f".{self.final_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self.tmp_path = Path(tmp.name)
        self.f = tmp
        self.count = 0
        self._closed = False
        self.f.write(struct.pack("<I", 0))  # patched on close

    def add(self, fen: str, wdl: tuple[float, float, float]) -> None:
        w, d, l_ = wdl
        b = fen.encode("utf-8")
        if len(b) > 0xFFFF:
            raise ValueError("FEN too long for u16 length field")
        self.f.write(struct.pack("<H", len(b)))
        self.f.write(b)
        self.f.write(struct.pack("<fff", w, d, l_))
        self.count += 1

    def commit(self) -> None:
        if self._closed:
            return
        self.f.seek(0)
        self.f.write(struct.pack("<I", self.count))
        self.f.flush()
        self.f.close()
        self.tmp_path.replace(self.final_path)
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self.f.close()
        self.tmp_path.unlink(missing_ok=True)
        self._closed = True


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def row_to_example(row: tuple[object, object, object, object], cp_scale: float) -> tuple[str, tuple[float, float, float]] | None:
    fen, cp, _mate, _depth = row
    if not isinstance(fen, str):
        return None
    if len(fen.encode("utf-8")) > 0xFFFF:
        return None
    try:
        cpf = float(cp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cpf):
        return None
    if not quiet_position(fen):
        return None
    return fen, cp_to_wdl(cpf, scale=cp_scale)


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
    ap.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 64)))
    ap.add_argument("--progress-every", type=int, default=5, help="Print progress every N fetched chunks.")
    args = ap.parse_args()

    if args.cp_scale <= 0:
        raise ValueError("--cp-scale must be > 0")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val-ratio must be in [0, 1]")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.min_depth < 0:
        raise ValueError("--min-depth must be >= 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    rng = random.Random(args.seed)
    stats = Stats()
    train = BinpackWriter(args.train_out)
    val = BinpackWriter(args.val_out)

    con = duckdb.connect(database=":memory:")
    limit_sql = "LIMIT ?" if args.max_rows > 0 else ""
    q = f"""
        SELECT fen, cp, mate, depth
        FROM read_parquet(?)
        WHERE depth >= ?
          AND mate IS NULL
          AND cp IS NOT NULL
          AND abs(cp) <= ?
        {limit_sql}
    """
    params: list[object] = [args.parquet.as_posix(), int(args.min_depth), float(args.cp_cap)]
    if args.max_rows > 0:
        params.append(int(args.max_rows))
    total_sql_rows = con.execute("SELECT COUNT(*) FROM (" + q + ")", params).fetchone()[0]
    con.execute(q, params)
    pool: concurrent.futures.ProcessPoolExecutor | None = None
    if args.workers > 1:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    started = time.time()
    chunk_idx = 0
    print(
        f"[dataprep] start workers={args.workers} total_sql_rows={int(total_sql_rows)} "
        f"chunk_size={args.chunk_size}"
    )

    try:
        while True:
            chunk = con.fetchmany(args.chunk_size)
            if not chunk:
                break
            chunk_idx += 1
            stats.passed_sql += len(chunk)
            iterator = (
                pool.map(row_to_example, chunk, [args.cp_scale] * len(chunk), chunksize=512)
                if pool is not None
                else (row_to_example(row, args.cp_scale) for row in chunk)
            )
            for example in iterator:
                stats.scanned_rows += 1
                if example is None:
                    continue
                fen, wdl = example
                stats.passed_quiet += 1
                if rng.random() < args.val_ratio:
                    val.add(fen, wdl)
                    stats.val_rows += 1
                else:
                    train.add(fen, wdl)
                    stats.train_rows += 1
            if chunk_idx % args.progress_every == 0:
                elapsed = max(1e-6, time.time() - started)
                scanned = stats.scanned_rows
                rows_per_sec = scanned / elapsed
                pct = (100.0 * scanned / total_sql_rows) if total_sql_rows > 0 else 0.0
                rem_rows = max(0, total_sql_rows - scanned)
                eta_sec = int(rem_rows / rows_per_sec) if rows_per_sec > 1e-9 else -1
                eta_txt = f"{eta_sec}s" if eta_sec >= 0 else "unknown"
                print(
                    f"[dataprep] chunk={chunk_idx} scanned={scanned}/{int(total_sql_rows)} "
                    f"({pct:.2f}%) quiet={stats.passed_quiet} train={stats.train_rows} "
                    f"val={stats.val_rows} speed={rows_per_sec:.0f} rows/s eta={eta_txt}"
                )
    except Exception:
        train.abort()
        val.abort()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        con.close()
        raise

    train.commit()
    val.commit()
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
    con.close()
    atomic_write_text(args.summary_out, json.dumps(asdict(stats), indent=2) + "\n")
    print(f"train={stats.train_rows} val={stats.val_rows} summary={args.summary_out}")


if __name__ == "__main__":
    main()
