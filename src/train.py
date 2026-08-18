"""
Training script -- reproduces the model from scratch (satisfies KLA's
GitHub requirement #3: "Python script or notebook that reproduces your
training process from scratch").

Usage:
    python train.py --data_root C:/semicon/train --epochs 100 --batch_size 16 \
        --device cuda --out_dir C:/semicon/checkpoints

On Colab, point --out_dir straight at a mounted Drive folder so best.pt/
last.pt never live only in the ephemeral /content filesystem, e.g.:
    python train.py --data_root /content/train --out_dir \
        /content/drive/MyDrive/kla_ps01/checkpoints --device cuda --use_perceptual

Training also writes resume.pt (model + optimizer + scheduler + epoch +
best_psnr) to --out_dir after every epoch, so a dropped Colab runtime can
pick back up with --resume instead of restarting from scratch. resume.pt
is separate from best.pt/last.pt, which stay plain state_dicts because
inference.py (and KLA's benchmarking harness) load them as-is.

Run with --smoke_test for a tiny few-step run to sanity check the full
pipeline before committing to a long run on a real GPU.
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import make_splits, SubsetRestorationDataset
from model import build_model
from losses import CombinedLoss


def psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((data_range ** 2) / mse)


def evaluate(model, loader, device):
    model.eval()
    psnr_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy_lr"].to(device)
            gt = batch["gt"].to(device)
            pred = model(noisy).clamp(0, 1)
            for i in range(pred.shape[0]):
                psnr_sum += psnr(pred[i], gt[i])
                n += 1
    model.train()
    return float(psnr_sum / max(n, 1))


def update_ema(ema_model, model, decay):
    """In-place EMA update of ema_model's params/buffers towards model's.
    A plateaued raw-weight trajectory (noisy near a minimum) is exactly the
    case EMA helps most -- averaging smooths that noise out for free."""
    with torch.no_grad():
        for ema_v, v in zip(ema_model.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                ema_v.copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="C:/semicon/train", help="folder containing GT/ and NoisyLR/")
    ap.add_argument("--out_dir", default="C:/semicon/checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_perceptual", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true", help="run a handful of steps only, to sanity check the pipeline")
    ap.add_argument("--resume", default=None, help="path to a resume.pt checkpoint to continue training from")
    ap.add_argument("--ckpt_every", type=int, default=1, help="save resume.pt every N epochs")

    ap.add_argument("--crop_size", type=int, default=96,
                     help="random crop size on the 128x128 noisy input during training (0 disables)")

    ap.add_argument("--disable_ema", action="store_true", help="turn off EMA weight averaging")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--lr_schedule", choices=["cosine", "cosine_warm_restarts"], default="cosine_warm_restarts")
    ap.add_argument("--restart_period", type=int, default=20, help="T_0 for cosine_warm_restarts")
    ap.add_argument("--restart_mult", type=int, default=2, help="T_mult for cosine_warm_restarts")
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--enc_blocks", default="2,2", help="comma-separated encoder block counts")
    ap.add_argument("--middle_blocks", type=int, default=4)
    ap.add_argument("--dec_blocks", default="2,2", help="comma-separated decoder block counts")

    ap.add_argument("--amp", action="store_true",
                     help="mixed-precision training via bfloat16 autocast. Unlike the fp16 AMP "
                          "that was reverted earlier (GradScaler skipped every step from fp16 "
                          "overflow on this dataset's out-of-range speckle-noise pixels), bf16 "
                          "shares fp32's 8-bit exponent range so it doesn't need a GradScaler "
                          "and doesn't overflow the same way -- see PROGRESS.md for the post-mortem")

    args = ap.parse_args()
    args.enc_blocks = tuple(int(x) for x in args.enc_blocks.split(","))
    args.dec_blocks = tuple(int(x) for x in args.dec_blocks.split(","))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    gt_dir = os.path.join(args.data_root, "GT")
    noisy_dir = os.path.join(args.data_root, "NoisyLR")

    crop_size = args.crop_size if args.crop_size > 0 else None

    train_ids, val_ids = make_splits(gt_dir, noisy_dir, val_fraction=args.val_fraction, seed=args.seed)
    train_ds = SubsetRestorationDataset(gt_dir, noisy_dir, train_ids, train=True, crop_size=crop_size)
    val_ds = SubsetRestorationDataset(gt_dir, noisy_dir, val_ids, train=False)

    if args.smoke_test:
        train_ds.gt_paths = train_ds.gt_paths[:8]
        val_ds.gt_paths = val_ds.gt_paths[:4]
        args.epochs = 1
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    device = torch.device(args.device)
    model = build_model(width=args.width, enc_blocks=args.enc_blocks,
                         middle_blocks=args.middle_blocks, dec_blocks=args.dec_blocks).to(device)
    loss_fn = CombinedLoss(use_perceptual=args.use_perceptual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.lr_schedule == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.restart_period, T_mult=args.restart_mult, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    use_ema = not args.disable_ema
    ema_model = None
    if use_ema:
        ema_model = build_model(width=args.width, enc_blocks=args.enc_blocks,
                                 middle_blocks=args.middle_blocks, dec_blocks=args.dec_blocks).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    print(f"train={len(train_ds)} val={len(val_ds)} device={device} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"ema={use_ema} crop_size={crop_size} lr_schedule={args.lr_schedule} amp_bf16={args.amp}")

    best_psnr = -1.0
    start_epoch = 0
    max_steps = 3 if args.smoke_test else None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_psnr = ckpt["best_psnr"]
        start_epoch = ckpt["epoch"] + 1
        if use_ema:
            if "ema_model" in ckpt:
                ema_model.load_state_dict(ckpt["ema_model"])
            else:
                ema_model.load_state_dict(model.state_dict())
        print(f"resumed from {args.resume} at epoch {start_epoch} (best_psnr={best_psnr:.2f}dB)")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = {}
        step = 0
        for batch in train_loader:
            noisy = batch["noisy_lr"].to(device)
            gt = batch["gt"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                pred = model(noisy)
                loss, logs = loss_fn(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_ema:
                update_ema(ema_model, model, args.ema_decay)

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            step += 1
            if max_steps and step >= max_steps:
                break

        scheduler.step()
        avg = {k: v / step for k, v in running.items()}
        val_psnr = evaluate(model, val_loader, device)
        ema_val_psnr = evaluate(ema_model, val_loader, device) if use_ema else -1.0
        dt = time.time() - t0

        ema_str = f" (ema {ema_val_psnr:.2f}dB)" if use_ema else ""
        print(f"epoch {epoch+1}/{args.epochs} | loss {avg['total']:.4f} "
              f"(charb {avg['charbonnier']:.4f} ssim {avg['ssim_loss']:.4f}) "
              f"| val_psnr {val_psnr:.2f}dB{ema_str} | {dt:.1f}s")

        # best.pt tracks whichever of raw/EMA weights scores higher on val --
        # EMA usually wins once training has plateaued, raw weights can still
        # win early on before the average has caught up.
        cur_best = max(val_psnr, ema_val_psnr)
        if cur_best > best_psnr:
            best_psnr = cur_best
            best_state = ema_model.state_dict() if ema_val_psnr >= val_psnr else model.state_dict()
            torch.save(best_state, os.path.join(args.out_dir, "best.pt"))

        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
            }
            if use_ema:
                ckpt["ema_model"] = ema_model.state_dict()
            torch.save(ckpt, os.path.join(args.out_dir, "resume.pt"))

    print(f"done. best val_psnr={best_psnr:.2f}dB, weights in {args.out_dir}")


if __name__ == "__main__":
    main()
