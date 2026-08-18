# KLA PS01 — AI-Based Restoration of Degraded Semiconductor Inspection Images

Single model that simultaneously removes speckle noise, removes Gaussian
noise, and 2x-upscales degraded semiconductor inspection images
(grayscale, 128→256 or 256→512), trained on paired KLA data.

## Setup

```bash
pip install -r requirements.txt
```

GPU is optional for inference (works on CPU, just slower) but required for
training in reasonable time.

## Quick usage example

Clone the repo and run inference on the bundled sample images — the
trained weights (`checkpoints/best.pt`) are already included, so this
works right away with no training required:

```bash
git clone https://github.com/Ganesh-0509/semicon.git
cd semicon
pip install -r requirements.txt

python src/inference.py NoisyLR/ restored_out/
```

This restores every `.npy` in `NoisyLR/` (400 sample degraded images) and
writes the denoised, 2x-upscaled result to `restored_out/`, one `.npy`
per input, plus a per-run average inference time. Swap `NoisyLR/` for
any folder of your own degraded `.npy`/`.png`/`.tif`/`.jpg` images.

## Repository layout

```
src/
  dataset.py     # loads paired GT/NoisyLR .npy files, augmentation (incl. random crop)
  model.py       # NAFNet-lite restoration network + fused 2x SR head, configurable capacity
  losses.py      # Charbonnier + SSIM + multi-layer VGG perceptual + FFT + Sobel edge loss
  train.py       # training from scratch — EMA, warm-restart LR, capacity flags
  inference.py   # KLA's evaluation entrypoint — this is the file KLA runs, self-ensemble TTA available
checkpoints/
  best.pt        # trained weights (highest val PSNR; width=64, 2.98M params)
sample_outputs/  # this model's restored output on the 400 bundled NoisyLR/ test images
requirements.txt
```

## Training (from scratch)

```bash
python src/train.py --data_root <path_to>/train --epochs 100 --batch_size 16 --device cuda
```

`<path_to>/train` must contain `GT/` and `NoisyLR/` subfolders of matching
`.npy` files (as delivered by KLA). Best checkpoint (by validation PSNR) is
saved to `checkpoints/best.pt`.

Add `--smoke_test` to run a few steps on a handful of samples first, to
verify the environment is set up correctly before a full run.

## Inference / Evaluation

```bash
python src/inference.py <input_dir> <output_dir>
```

- `<input_dir>`: folder of degraded `.npy` (or `.png`/`.tif`/`.jpg`) test images.
- `<output_dir>`: created automatically if it doesn't exist. One restored
  `.npy` file per input is written here, same filename, float32 in [0, 1],
  same format as the ground-truth files — drop-in compatible with a
  standard SSIM/PSNR/LPIPS scorer.
- Loads weights from `checkpoints/best.pt` by default; override with
  `--weights <path>` if needed. Device auto-selects CUDA if available,
  otherwise CPU (`--device` to force).
- Prints per-run average/min/max inference time in ms/image at the end.

This script requires no manual edits to run — it accepts the input/output
directories as either positional or named (`--input_dir`/`--output_dir`)
arguments.

## Approach

- **Architecture:** NAFNet-lite (Chen et al. 2022 style block: SimpleGate +
  simplified channel attention, no self-attention, no GAN) with a fused
  PixelShuffle 2x super-resolution head, so denoising and upscaling happen
  in a single forward pass. Chosen over SwinIR/Restormer for its
  quality-per-FLOP — the problem statement explicitly benchmarks and
  penalizes slow inference on the grading H100.
- **Loss:** Charbonnier (robust pixel fidelity) + SSIM (matches a grading
  metric directly) + multi-layer VGG perceptual (relu1_2/relu2_2/relu3_3,
  helps LPIPS without a GAN's artifact/ringing risk) + FFT-magnitude loss
  (targets high-frequency detail the spatial losses under-penalize) +
  Sobel edge loss (explicit edge-boundary supervision).
- **Generalization:** training data is augmented with randomized *extra*
  speckle/Gaussian noise on top of KLA's provided degradation (not just
  geometric flips/rotations), plus random spatial crops, so the model
  doesn't memorize KLA's fixed noise recipe — this targets the explicit
  out-of-distribution grading criterion.
- **Training stability:** EMA weight averaging and a cosine warm-restart
  LR schedule, both added after the training curve was observed to
  plateau on an earlier, smaller configuration.
- **Inference-time note:** self-ensemble TTA (`--tta flip2`/`flip8`) is
  available in `inference.py` but **not** the default — measured directly
  against this checkpoint's own validation split, TTA improves PSNR/SSIM
  slightly but makes LPIPS measurably *worse* (image averaging smooths
  away the fine texture LPIPS rewards preserving), so the default
  (`--tta none`) is the one actually optimized for the metric most tied
  to "does this look structurally real."

## Notes for reviewers

- Dataset format observed directly from the delivered files: GT is
  256x256 float32 in [0, 1]; NoisyLR is 128x128 float32 and *can exceed*
  [0, 1] (speckle noise pushes values past the true range — this is
  expected and the model is trained on the raw unclipped values
  accordingly).
- `requirements.txt` reflects a CPU dev environment used for pipeline
  validation; re-run `pip freeze > requirements.txt` after final training
  on the actual GPU environment before submission.
