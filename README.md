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

## Repository layout

```
src/
  dataset.py     # loads paired GT/NoisyLR .npy files, augmentation
  model.py       # NAFNet-lite restoration network + fused 2x SR head
  losses.py      # Charbonnier + SSIM + light VGG-perceptual loss
  train.py       # training from scratch
  inference.py   # KLA's evaluation entrypoint — this is the file KLA runs
checkpoints/
  best.pt        # trained weights (highest val PSNR)
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
  metric directly) + a small-weight VGG perceptual term (helps LPIPS
  without a GAN's artifact/ringing risk, which the problem statement
  explicitly warns against).
- **Generalization:** training data is augmented with randomized *extra*
  speckle/Gaussian noise on top of KLA's provided degradation (not just
  geometric flips/rotations), so the model doesn't memorize KLA's fixed
  noise recipe — this targets the explicit out-of-distribution grading
  criterion.

## Notes for reviewers

- Dataset format observed directly from the delivered files: GT is
  256x256 float32 in [0, 1]; NoisyLR is 128x128 float32 and *can exceed*
  [0, 1] (speckle noise pushes values past the true range — this is
  expected and the model is trained on the raw unclipped values
  accordingly).
- `requirements.txt` reflects a CPU dev environment used for pipeline
  validation; re-run `pip freeze > requirements.txt` after final training
  on the actual GPU environment before submission.
