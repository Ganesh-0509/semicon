"""
KLA-spec evaluation/inference script.

Per the problem statement, this file "will be used AS-IS by KLA's
benchmarking team ... on the H100 GPU. If your script does not run
without manual edits, your submission cannot be benchmarked."
So: no hardcoded absolute paths, no notebook-only state, sane defaults,
auto-creates the output directory, and works whether invoked with
positional args or named flags (the exact CLI contract KLA will use
wasn't specified beyond "accepts a path to the test images directory
and a path to the output directory").

Usage:
    python inference.py <input_dir> <output_dir>
    python inference.py --input_dir <input_dir> --output_dir <output_dir> --weights <path/to/best.pt>

Defaults to --tta flip2 (predict on the input and its h-flip, average the
two, 2x inference cost) for a small free PSNR/SSIM/LPIPS bump. Pass
--tta none if KLA's H100 latency budget can't absorb that, or --tta flip8
for the full 8-way dihedral self-ensemble if it can absorb more.

Input format: .npy files, float32, 128x128 (or 256x256 for the 256->512
scale case), grayscale, matching the format KLA's own training data was
delivered in. .png/.tif/.jpg inputs are also accepted as a fallback.
Output: one .npy file per input, float32 in [0, 1], upscaled 2x, written
to <output_dir>/<same_filename>.npy -- same format as the GT files, so
it drops straight into their SSIM/PSNR/LPIPS scorer.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "best.pt")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Self-ensemble transform sets: (k, flip) = rotate 90*k degrees, then h-flip if flip.
# "flip2" is the cheap default (2x cost); "flip8" is the full dihedral group (8x cost).
_TTA_TRANSFORMS = {
    "none": [(0, False)],
    "flip2": [(0, False), (0, True)],
    "flip8": [(k, f) for k in range(4) for f in (False, True)],
}


def _apply_transform(x, k, flip):
    if flip:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def _invert_transform(x, k, flip):
    if k:
        x = torch.rot90(x, -k, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


def run_with_tta(model, x, tta):
    """Averages predictions over geometric self-ensemble transforms -- the
    model was already trained to be equivariant to flips/rotations (see
    dataset.py's augmentation), so this squeezes out a consistent PSNR/SSIM
    gain for free, at (num transforms)x inference cost."""
    preds = []
    for k, flip in _TTA_TRANSFORMS[tta]:
        xt = _apply_transform(x, k, flip)
        pt = model(xt)
        preds.append(_invert_transform(pt, k, flip))
    return torch.stack(preds, dim=0).mean(dim=0)


def load_input(path):
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    elif path.lower().endswith(IMAGE_EXTS):
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.asarray(img).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported input file type: {path}")
    return arr


def save_output(arr, out_path):
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    np.save(out_path, arr)


def parse_args():
    ap = argparse.ArgumentParser(description="Run KLA restoration model inference on a directory of test images.")
    ap.add_argument("input_dir", nargs="?", default=None, help="path to test images directory")
    ap.add_argument("output_dir", nargs="?", default=None, help="path to write restored outputs")
    ap.add_argument("--input_dir", dest="input_dir_flag", default=None)
    ap.add_argument("--output_dir", dest="output_dir_flag", default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to trained model weights (.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", choices=list(_TTA_TRANSFORMS), default="flip2",
                     help="geometric self-ensembling: none (fastest), flip2 (2x cost, default), flip8 (8x cost)")
    args = ap.parse_args()

    input_dir = args.input_dir_flag or args.input_dir
    output_dir = args.output_dir_flag or args.output_dir
    if not input_dir or not output_dir:
        ap.error("both an input directory and an output directory are required "
                  "(positionally or via --input_dir/--output_dir)")
    return input_dir, output_dir, args.weights, args.device, args.tta


def main():
    input_dir, output_dir, weights_path, device_str, tta = parse_args()

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str)
    model = build_model().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy") or f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        print(f"[inference] WARNING: no .npy/image files found in {input_dir}")
        return

    print(f"[inference] {len(files)} files | device={device} | weights={weights_path} | tta={tta}")

    times = []
    with torch.no_grad():
        for fname in files:
            in_path = os.path.join(input_dir, fname)
            arr = load_input(in_path)

            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW

            t0 = time.time()
            pred = run_with_tta(model, x, tta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(dt)

            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
            out_name = os.path.splitext(fname)[0] + ".npy"
            save_output(pred_np, os.path.join(output_dir, out_name))

    avg_ms = 1000 * sum(times) / len(times)
    print(f"[inference] done. {len(files)} images -> {output_dir}")
    print(f"[inference] avg inference time: {avg_ms:.1f} ms/image "
          f"(min {1000*min(times):.1f} ms, max {1000*max(times):.1f} ms)")


if __name__ == "__main__":
    main()
