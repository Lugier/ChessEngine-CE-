#!/usr/bin/env python3
"""
Train the small 768->256->1 Cortex NNUE head on packed data from data/prepare_binpack.py.
Uses Apple MPS when available, else CPU. No cloud.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
import torch.nn as nn


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TinyNnue(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(768, hidden)
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_binpack(path: Path):
    data = path.read_bytes()
    off = 0
    (n,) = struct.unpack_from("<I", data, off)
    off += 4
    fens: list[str] = []
    wdls: list[tuple[float, float, float]] = []
    for _ in range(n):
        (flen,) = struct.unpack_from("<H", data, off)
        off += 2
        fen = data[off : off + flen].decode("utf-8")
        off += flen
        w, d, l_ = struct.unpack_from("<fff", data, off)
        off += 12
        fens.append(fen)
        wdls.append((w, d, l_))
    return fens, wdls


def fen_to_features(fen: str) -> list[float]:
    # Must match engine nnue.cpp board_to_features (768 one-hot).
    MAP = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }
    board = fen.split()[0]
    rows = board.split("/")
    feat = [0.0] * 768
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                sq = (7 - r) * 8 + c
                p = MAP.get(ch)
                if p is not None:
                    feat[sq * 12 + p] = 1.0
                c += 1
    return feat


def export_cxn1(model: TinyNnue, path: Path, out_scale: int = 16) -> None:
    w1 = model.fc1.weight.detach().cpu().numpy().astype("int16")
    b1 = model.fc1.bias.detach().cpu().numpy().astype("int16")
    w2 = model.fc2.weight.detach().cpu().numpy().astype("int16")
    b2 = model.fc2.bias.detach().cpu().numpy().astype("int16")
    # Engine uses single scalar from hidden; fold WDL head to first output.
    w2_row = w2[0].copy()
    b2_scalar = int(b2[0])
    buf = bytearray()
    buf += b"CXN1"
    buf += struct.pack("<I", 1)
    buf += struct.pack("<i", out_scale)
    buf += w1.tobytes()
    buf += b1.tobytes()
    buf += w2_row.tobytes()
    buf += struct.pack("<h", b2_scalar)
    path.write_bytes(buf)
    print(f"exported {path} ({len(buf)} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("cortex.nnue"))
    args = ap.parse_args()

    fens, wdls = load_binpack(args.data)
    if not fens:
        raise SystemExit("no training rows")
    X = torch.tensor([fen_to_features(f) for f in fens], dtype=torch.float32)
    y = torch.tensor(wdls, dtype=torch.float32)
    dev = pick_device()
    X, y = X.to(dev), y.to(dev)

    model = TinyNnue().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"epoch {ep} loss {loss.item():.6f}")

    export_cxn1(model.cpu(), args.out)


if __name__ == "__main__":
    main()
