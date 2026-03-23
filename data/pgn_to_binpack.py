#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import chess
import chess.pgn


def quiet_position(board: chess.Board) -> bool:
    if board.is_check():
        return False
    for mv in board.legal_moves:
        if board.is_capture(mv) or mv.promotion is not None:
            return False
    return True


def result_to_wdl(result: str) -> tuple[float, float, float]:
    if result == "1-0":
        return (0.94, 0.04, 0.02)
    if result == "0-1":
        return (0.02, 0.04, 0.94)
    return (0.08, 0.84, 0.08)


def blend_target(base: tuple[float, float, float], ply: int) -> tuple[float, float, float]:
    neutral = (0.33, 0.34, 0.33)
    w = min(1.0, max(0.35, ply / 80.0))
    out = tuple(w * b + (1.0 - w) * n for b, n in zip(base, neutral))
    s = sum(out)
    return (out[0] / s, out[1] / s, out[2] / s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=140)
    ap.add_argument("--ply-step", type=int, default=2)
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()

    if args.ply_step <= 0:
        raise SystemExit("--ply-step must be > 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.pgn.open("r", encoding="utf-8", errors="replace") as fin, args.out.open("wb") as fout:
        fout.write(struct.pack("<I", 0))
        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            result = game.headers.get("Result", "1/2-1/2")
            base_wdl = result_to_wdl(result)
            board = game.board()
            ply = 0
            for mv in game.mainline_moves():
                board.push(mv)
                ply += 1
                if ply < args.min_ply or ply > args.max_ply or (ply % args.ply_step) != 0:
                    continue
                if not quiet_position(board):
                    continue
                fen = board.fen()
                fen_b = fen.encode("utf-8")
                if len(fen_b) > 0xFFFF:
                    continue
                w, d, l_ = blend_target(base_wdl, ply)
                if not (math.isfinite(w) and math.isfinite(d) and math.isfinite(l_)):
                    continue
                fout.write(struct.pack("<H", len(fen_b)))
                fout.write(fen_b)
                fout.write(struct.pack("<fff", w, d, l_))
                count += 1
                if args.max_positions > 0 and count >= args.max_positions:
                    break
            if args.max_positions > 0 and count >= args.max_positions:
                break

        fout.seek(0)
        fout.write(struct.pack("<I", count))
    print(f"wrote {count} positions to {args.out}")


if __name__ == "__main__":
    main()
