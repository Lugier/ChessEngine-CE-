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
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        one_m = 1.0 - self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=one_m)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def load_binpack(path: Path, max_rows: int = 0):
    if max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")
    limit = None if max_rows == 0 else max_rows
    fens: list[str] = []
    wdls: list[tuple[float, float, float]] = []
    with path.open("rb") as f:
        hdr = f.read(4)
        if len(hdr) != 4:
            raise SystemExit(f"invalid binpack: too short ({path})")
        (n_total,) = struct.unpack("<I", hdr)
        for _ in range(n_total):
            if limit is not None and len(fens) >= limit:
                break
            flen_b = f.read(2)
            if len(flen_b) != 2:
                raise SystemExit(f"invalid binpack: truncated flen ({path})")
            (flen,) = struct.unpack("<H", flen_b)
            fen_b = f.read(flen)
            tgt_b = f.read(12)
            if len(fen_b) != flen or len(tgt_b) != 12:
                raise SystemExit(f"invalid binpack: truncated record ({path})")
            fen = fen_b.decode("utf-8")
            w, d, l_ = struct.unpack("<fff", tgt_b)
            if not (math.isfinite(w) and math.isfinite(d) and math.isfinite(l_)):
                continue
            fens.append(fen)
            wdls.append((w, d, l_))
    return fens, wdls


class FenDataset(Dataset):
    def __init__(self, fens: list[str], wdls: list[tuple[float, float, float]], to_feat):
        self.fens = fens
        self.wdls = wdls
        self.to_feat = to_feat

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int):
        feat = torch.tensor(self.to_feat(self.fens[idx]), dtype=torch.float32)
        tgt = torch.tensor(self.wdls[idx], dtype=torch.float32)
        return feat, tgt


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
    # Engine uses single scalar from hidden; fold WDL head as (W - L).
    w2_i32 = w2.astype("int32")
    b2_i32 = b2.astype("int32")
    w2_row32 = w2_i32[0] - w2_i32[2]
    b2_scalar32 = int(b2_i32[0] - b2_i32[2])
    maxv = max(int(abs(w2_row32).max()), abs(b2_scalar32), 1)
    k = max(1, (maxv + 32766) // 32767)
    w2_row = (w2_row32 // k).astype("int16")
    b2_scalar = int(max(-32767, min(32767, b2_scalar32 // k)))
    buf = bytearray()
    buf += b"CXN1" if version == 1 else b"CXN2"
    buf += struct.pack("<I", version)
    eff_out_scale = max(1, int(out_scale) * int(s1) * int(s2) * int(k))
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
    ap.add_argument("--aux-data", type=Path, default=None, help="Optional self-play binpack mixed into train set.")
    ap.add_argument("--aux-ratio", type=float, default=0.0, help="Fraction of train rows to sample from --aux-data.")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=1000000)
    ap.add_argument("--val-max-rows", type=int, default=250000)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--loss", choices=["mse", "kldiv"], default="kldiv")
    ap.add_argument("--label-smoothing", type=float, default=0.01)
    ap.add_argument("--wdl-temperature", type=float, default=1.2)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--use-ema", type=int, default=1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--cpu-interop-threads", type=int, default=1)
    ap.add_argument(
        "--feature-mode",
        choices=["legacy", "kingbucket"],
        default="legacy",
        help="legacy=768 (CXN1), kingbucket=10240 HalfKP-like (CXN2).",
    )
    args = ap.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if args.log_interval <= 0:
        raise SystemExit("--log-interval must be > 0")
    if args.prefetch_factor <= 0:
        raise SystemExit("--prefetch-factor must be > 0")
    if args.val_every <= 0:
        raise SystemExit("--val-every must be > 0")
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(max(1, args.cpu_interop_threads))

    fens, wdls = load_binpack(args.data, max_rows=args.max_rows)
    if not fens:
        raise SystemExit("no training rows")
    if args.aux_data is not None and args.aux_data.exists() and args.aux_ratio > 0:
        aux_cap = int(len(fens) * max(0.0, min(1.0, args.aux_ratio)))
        if aux_cap > 0:
            afens, awdls = load_binpack(args.aux_data, max_rows=aux_cap)
            if afens:
                fens.extend(afens)
                wdls.extend(awdls)
    to_feat = fen_to_features_legacy if args.feature_mode == "legacy" else fen_to_features_kingbucket
    input_dim = 768 if args.feature_mode == "legacy" else 16 * 64 * 10
    version = 1 if args.feature_mode == "legacy" else 2
    train_ds = FenDataset(fens, wdls, to_feat)
    val_ds = None
    if args.val_data is not None and args.val_data.exists():
        vfens, vwdls = load_binpack(args.val_data, max_rows=args.val_max_rows)
        if vfens:
            val_ds = FenDataset(vfens, vwdls, to_feat)
    dev = pick_device()
    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    pin = use_cuda
    workers = args.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=(args.prefetch_factor if workers > 0 else None),
    )
    val_loader = None
    if val_ds is not None:
        val_workers = min(workers, 8)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=pin,
            persistent_workers=val_workers > 0,
            prefetch_factor=(args.prefetch_factor if val_workers > 0 else None),
        )

    model = TinyNnue(input_dim=input_dim).to(dev)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_ema = bool(args.use_ema)
    ema = ModelEMA(model, decay=args.ema_decay) if use_ema else None
    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.KLDivLoss(reduction="batchmean")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and args.amp_dtype == "fp16")
    print(
        f"[train] device={dev.type} cuda={use_cuda} "
        f"train_rows={len(train_ds)} val_rows={len(val_ds) if val_ds is not None else 0} "
        f"batch_size={args.batch_size} workers={workers}"
    )
    start_ep = 0
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_ep = int(ck["epoch"]) + 1
        print(f"resumed from {args.resume} at epoch {start_ep}")

    warmup_steps = max(1, int(len(train_loader) * max(0.0, args.warmup_ratio) * max(1, args.epochs)))
    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    global_step = 0

    def build_targets(yb: torch.Tensor) -> torch.Tensor:
        y = yb.clamp(1e-6, 1.0)
        if args.wdl_temperature != 1.0:
            y = y.pow(1.0 / args.wdl_temperature)
            y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if args.label_smoothing > 0:
            eps = max(0.0, min(0.2, args.label_smoothing))
            y = y * (1.0 - eps) + eps / 3.0
            y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return y

    def step_scheduler() -> None:
        nonlocal global_step
        global_step += 1
        if args.scheduler == "none":
            return
        if global_step <= warmup_steps:
            lr_mult = global_step / warmup_steps
        else:
            p = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * p))
            lr_mult = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_mult

    for ep in range(start_ep, args.epochs):
        ep_start = time.time()
        model.train()
        loss_sum = 0.0
        seen = 0
        num_batches = max(1, len(train_loader))
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(dev, non_blocking=pin)
            yb = yb.to(dev, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            yb_proc = build_targets(yb)
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    pred = model(xb)
                    loss = loss_fn(pred, yb_proc) if args.loss == "mse" else loss_fn(torch.log_softmax(pred, dim=1), yb_proc)
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb_proc) if args.loss == "mse" else loss_fn(torch.log_softmax(pred, dim=1), yb_proc)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                opt.step()
            step_scheduler()
            if ema is not None:
                ema.update(model)
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            seen += bs
            if step % args.log_interval == 0 or step == num_batches:
                elapsed = max(1e-6, time.time() - ep_start)
                bps = step / elapsed
                ep_pct = 100.0 * step / num_batches
                eta_step = int((num_batches - step) / max(1e-6, bps))
                print(
                    f"[train-step] epoch={ep + 1}/{args.epochs} "
                    f"step={step}/{num_batches} ({ep_pct:.1f}%) "
                    f"loss={float(loss.item()):.6f} "
                    f"samples={seen} steps_per_s={bps:.2f} eta={eta_step}s"
                )
        avg_loss = loss_sum / max(1, seen)
        val_txt = ""
        if val_loader is not None and (ep % args.val_every == 0 or ep + 1 == args.epochs):
            model.eval()
            vloss_sum = 0.0
            vseen = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev, non_blocking=pin)
                    yb = yb.to(dev, non_blocking=pin)
                    yb_proc = build_targets(yb)
                    if use_cuda:
                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                            vpred = model(xb)
                            vloss = loss_fn(vpred, yb_proc) if args.loss == "mse" else loss_fn(torch.log_softmax(vpred, dim=1), yb_proc)
                    else:
                        vpred = model(xb)
                        vloss = loss_fn(vpred, yb_proc) if args.loss == "mse" else loss_fn(torch.log_softmax(vpred, dim=1), yb_proc)
                    bs = xb.shape[0]
                    vloss_sum += float(vloss.item()) * bs
                    vseen += bs
            val_txt = f" val_loss={vloss_sum / max(1, vseen):.6f}"
        elapsed_ep = max(1e-6, time.time() - ep_start)
        done = (ep - start_ep + 1)
        total = max(1, args.epochs - start_ep)
        pct = 100.0 * done / total
        eta = int((total - done) * elapsed_ep)
        print(
            f"[train] epoch={ep + 1}/{args.epochs} ({pct:.1f}%) "
            f"loss={avg_loss:.6f}{val_txt} epoch_time={elapsed_ep:.1f}s eta={eta}s"
        )
        if args.checkpoint_dir is not None and args.save_every > 0 and (ep + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint_dir / f"epoch-{ep+1:04d}.pt", model, opt, ep)

    if ema is not None:
        ema.copy_to(model)
    export_cxn(model.cpu(), args.out, version=version)


if __name__ == "__main__":
    main()
