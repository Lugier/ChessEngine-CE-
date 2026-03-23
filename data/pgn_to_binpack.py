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


def tactical_position(board: chess.Board) -> bool:
    if board.is_check():
        return True
    checking = 0
    for mv in board.legal_moves:
        if board.is_capture(mv) or mv.promotion is not None:
            return True
        if board.gives_check(mv):
            checking += 1
            if checking >= 2:
                return True
    return False


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
    ap.add_argument("--strict-legal", type=int, default=1, help="Fail on illegal moves in PGN stream (1=yes).")
    ap.add_argument("--min-output-positions", type=int, default=1000, help="Fail if fewer positions are produced.")
    ap.add_argument(
        "--position-filter",
        choices=["all", "quiet", "tactical"],
        default="all",
        help="Select which position types to keep.",
    )
    args = ap.parse_args()

    if args.ply_step <= 0:
        raise SystemExit("--ply-step must be > 0")
    if args.min_output_positions < 0:
        raise SystemExit("--min-output-positions must be >= 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    games_total = 0
    games_illegal = 0
    skipped_filter = 0
    skipped_nonfinite = 0
    skipped_fen_len = 0
    with args.pgn.open("r", encoding="utf-8", errors="replace") as fin, args.out.open("wb") as fout:
        fout.write(struct.pack("<I", 0))
        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            games_total += 1
            result = game.headers.get("Result", "1/2-1/2")
            base_wdl = result_to_wdl(result)
            board = game.board()
            ply = 0
            game_illegal = False
            for mv in game.mainline_moves():
                if args.strict_legal == 1 and mv not in board.legal_moves:
                    game_illegal = True
                    break
                board.push(mv)
                ply += 1
                if ply < args.min_ply or ply > args.max_ply or (ply % args.ply_step) != 0:
                    continue
                if args.position_filter == "quiet" and not quiet_position(board):
                    skipped_filter += 1
                    continue
                if args.position_filter == "tactical" and not tactical_position(board):
                    skipped_filter += 1
                    continue
                fen = board.fen()
                fen_b = fen.encode("utf-8")
                if len(fen_b) > 0xFFFF:
                    skipped_fen_len += 1
                    continue
                w, d, l_ = blend_target(base_wdl, ply)
                if not (math.isfinite(w) and math.isfinite(d) and math.isfinite(l_)):
                    skipped_nonfinite += 1
                    continue
                fout.write(struct.pack("<H", len(fen_b)))
                fout.write(fen_b)
                fout.write(struct.pack("<fff", w, d, l_))
                count += 1
                if args.max_positions > 0 and count >= args.max_positions:
                    break
            if game_illegal:
                games_illegal += 1
                if args.strict_legal == 1:
                    raise SystemExit(f"illegal move detected in PGN stream at game #{games_total}")
            if args.max_positions > 0 and count >= args.max_positions:
                break

        fout.seek(0)
        fout.write(struct.pack("<I", count))
    print(
        f"wrote {count} positions to {args.out} | games={games_total} illegal={games_illegal} "
        f"skipped_filter={skipped_filter} skipped_nonfinite={skipped_nonfinite} skipped_fen_len={skipped_fen_len}"
    )
    if count < args.min_output_positions:
        raise SystemExit(
            f"too few output positions: {count} < {args.min_output_positions} "
            f"(selfplay data quality gate)"
        )


if __name__ == "__main__":
    main()
