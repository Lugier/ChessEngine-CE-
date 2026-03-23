#!/usr/bin/env python3
"""
Supervised WDL training for the Cortex NNUE stack.

- Input: binpack from data/prepare_binpack.py (FEN + win/draw/loss targets).
- Model: 768 sparse one-hot (must match engine nnue.cpp) -> ReLU -> 256 -> 3 logits.
- Loss: MSE on the three WDL floats (Gemini: smooth target space vs raw cp).
- Export: CXN1 uses a single scalar head; we fold fc2 by taking output row 0 only
  (see export_cxn1). The engine multiplies hidden by w2[j] and adds b2 — dimensions
  must stay 256-wide for w2.

Device: MPS on Apple Silicon if available, else CPU. For large data, train on GPU
in the cloud is out of repo scope; the script stays lean for verify/smoke.
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
    def __init__(self, input_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
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


def fen_to_features_legacy(fen: str) -> list[float]:
    # Must match engine nnue.cpp board_to_features legacy mode.
    MAP = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
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


def fen_to_features_kingbucket(fen: str) -> list[float]:
    # HalfKP-like simplification: side-to-move king bucket (16) + piece-square planes.
    board_part, stm = fen.split()[:2]
    MAP = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "p": 5, "n": 6, "b": 7, "r": 8, "q": 9}
    rows = board_part.split("/")
    wk = bk = 0
    pieces: list[tuple[int, str]] = []
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
                continue
            sq = (7 - r) * 8 + c
            if ch == "K":
                wk = sq
            elif ch == "k":
                bk = sq
            else:
                pieces.append((sq, ch))
            c += 1
    ksq = wk if stm == "w" else bk
    bucket = (ksq % 8) // 2 + 4 * ((ksq // 8) // 2)
    feat = [0.0] * (16 * 64 * 10)
    for sq, ch in pieces:
        p = MAP.get(ch)
        if p is None:
            continue
        idx = bucket * (64 * 10) + sq * 10 + p
        feat[idx] = 1.0
    return feat


def export_cxn(model: TinyNnue, path: Path, out_scale: int = 16, version: int = 1) -> None:
    w1f = model.fc1.weight.detach().cpu().numpy()
    b1f = model.fc1.bias.detach().cpu().numpy()
    w2f = model.fc2.weight.detach().cpu().numpy()
    b2f = model.fc2.bias.detach().cpu().numpy()
    # Layer-wise int16 quantization with explicit scaling to avoid near-zero casts.
    max1 = max(float(abs(w1f).max()), float(abs(b1f).max()), 1e-8)
    max2 = max(float(abs(w2f).max()), float(abs(b2f).max()), 1e-8)
    s1 = min(256, max(1, int(30000.0 / max1)))
    s2 = min(256, max(1, int(30000.0 / max2)))
    w1 = (w1f * s1).round().clip(-32767, 32767).astype("int16")
    b1 = (b1f * s1).round().clip(-32767, 32767).astype("int16")
    w2 = (w2f * s2).round().clip(-32767, 32767).astype("int16")
    b2 = (b2f * s1 * s2).round().clip(-32767, 32767).astype("int16")
    # Engine uses single scalar from hidden; fold WDL head to first output.
    w2_row = w2[0].copy()
    b2_scalar = int(b2[0])
    buf = bytearray()
    buf += b"CXN1" if version == 1 else b"CXN2"
    buf += struct.pack("<I", version)
    eff_out_scale = max(1, int(out_scale) * int(s1) * int(s2))
    buf += struct.pack("<i", eff_out_scale)
    buf += w1.tobytes()
    buf += b1.tobytes()
    buf += w2_row.tobytes()
    buf += struct.pack("<h", b2_scalar)
    path.write_bytes(buf)
    print(f"exported v{version} {path} ({len(buf)} bytes) s1={s1} s2={s2} out_scale={eff_out_scale}")


def save_checkpoint(path: Path, model: TinyNnue, opt: torch.optim.Optimizer, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()},
        path,
    )
    print(f"checkpoint {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("cortex.nnue"))
    ap.add_argument("--val-data", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--feature-mode",
        choices=["legacy", "kingbucket"],
        default="legacy",
        help="legacy=768 (CXN1), kingbucket=10240 HalfKP-like (CXN2).",
    )
    args = ap.parse_args()

    fens, wdls = load_binpack(args.data)
    if not fens:
        raise SystemExit("no training rows")
    to_feat = fen_to_features_legacy if args.feature_mode == "legacy" else fen_to_features_kingbucket
    input_dim = 768 if args.feature_mode == "legacy" else 16 * 64 * 10
    version = 1 if args.feature_mode == "legacy" else 2
    X = torch.tensor([to_feat(f) for f in fens], dtype=torch.float32)
    y = torch.tensor(wdls, dtype=torch.float32)
    Xv = yv = None
    if args.val_data is not None and args.val_data.exists():
        vfens, vwdls = load_binpack(args.val_data)
        if vfens:
            Xv = torch.tensor([to_feat(f) for f in vfens], dtype=torch.float32)
            yv = torch.tensor(vwdls, dtype=torch.float32)
    dev = pick_device()
    X, y = X.to(dev), y.to(dev)
    if Xv is not None and yv is not None:
        Xv, yv = Xv.to(dev), yv.to(dev)

    model = TinyNnue(input_dim=input_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    start_ep = 0
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_ep = int(ck["epoch"]) + 1
        print(f"resumed from {args.resume} at epoch {start_ep}")

    for ep in range(start_ep, args.epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"epoch {ep} loss {loss.item():.6f}")
            if Xv is not None and yv is not None:
                with torch.no_grad():
                    vloss = loss_fn(model(Xv), yv).item()
                print(f"epoch {ep} val_loss {vloss:.6f}")
        if args.checkpoint_dir is not None and args.save_every > 0 and (ep + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint_dir / f"epoch-{ep+1:04d}.pt", model, opt, ep)

    export_cxn(model.cpu(), args.out, version=version)


if __name__ == "__main__":
    main()
