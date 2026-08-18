# KLA PS01 — Idea Submission Content
Team: [FILL IN] | File to save as: `TeamName_KLA_PS01.pdf` | Max 8-9 slides, remove instruction slide

> **Status note (delete before export):** Slide 6 below uses the **final shipped model
> (width=96, 28.83dB / 0.7870 / 0.2237)** — fully trained, verified end-to-end (fresh clone +
> `pip install` + `inference.py` tested from scratch), and live in `checkpoints/best.pt` on GitHub.
> An even larger 29.2M-param model reached the same PSNR but required Git LFS to ship (~117MB
> checkpoint) — deliberately not used, see Slide 5's innovation bullet on this tradeoff.

---

## Slide 1 — Team Details

- **Team Name:** [FILL IN]
- **Members & Roles:**
  - [Name 1] — [role, e.g. Model architecture & training]
  - [Name 2] — [role, e.g. Data pipeline & augmentation]
  - [Name 3] — [role, e.g. Evaluation, PPT, video]
  - [Name 4] — [role, e.g. GitHub/repro, documentation]
- **College:** [FILL IN]
- **Contact:** [email / phone]

---

## Slide 2 — Why This Problem Matters

**Problem selected:** AI-Based Restoration of Degraded Images (KLA)

In semiconductor manufacturing, every chip is verified using microscopic
inspection images. A single missed detail — one pixel of noise hiding a
defect, or a blur softening a hairline crack — can let a faulty chip pass
inspection and fail in the field, or cause a good chip to be scrapped
unnecessarily. As fabs push to sub-nanometer nodes, inspection systems
must examine more images faster, which often means capturing at lower
resolution or under noisier conditions to keep throughput up. That trade-off
is exactly what this problem asks us to reverse computationally: recover
a clean, full-resolution image from a fast, degraded capture — so
inspection speed and inspection quality stop being in tension with each
other. Getting restoration wrong in either direction is costly: over-smooth
and you erase real defects (false negatives, defective chips shipped);
under-restore and you leave noise that looks like a defect (false
positives, good chips scrapped). That's why the problem explicitly
penalizes both blur and hallucinated detail, not just raw noise.

---

## Slide 3 — Key Concept & Approach

- **Model type:** A compact convolutional encoder-decoder in the NAFNet
  family (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022)
  — no self-attention, no nonlinear activations, no adversarial training —
  with a fused PixelShuffle 2x super-resolution head bolted onto the
  decoder output.
- **Why this architecture:**
  - NAFNet-style blocks get near state-of-the-art denoising/deblurring
    quality at a fraction of the compute of transformer-based restoration
    models (SwinIR, Restormer) — directly relevant since KLA benchmarks
    and penalizes slow inference on their H100 grading run.
  - Deliberately **not** GAN-based (unlike Real-ESRGAN-style approaches):
    the problem statement explicitly warns against "artificial patterns or
    ringing," which is the classic GAN-restoration failure mode. We traded
    a small amount of texture sharpness for reliability and honesty of
    output.
- **How it addresses all 3 degradation types in one pass:**
  - *Speckle noise* → simplified channel-attention gating in each block
    learns to suppress high-variance pixel-level noise while preserving
    structure.
  - *Gaussian noise / haze* → the encoder-decoder's multi-scale receptive
    field restores edge sharpness without the ringing artifacts a naive
    sharpening filter would introduce.
  - *Resolution loss* → rather than treating super-resolution as a
    separate post-process, the PixelShuffle 2x head is fused directly onto
    the restoration decoder, so denoising and upscaling are learned
    jointly instead of compounding each other's errors in two passes.

---

## Slide 4 — Detailed Solution

**Architecture (NAFNet-lite + fused 2x SR head):**

```
Degraded input (1x128x128, unclipped float32)
        │
   Intro Conv (1→32ch)
        │
 ┌──────▼──────┐   skip ──────────────┐
 │ Encoder x2  │──Down(32→64)──┐      │
 └─────────────┘                │      │
                          ┌──────▼──────┐  skip ──┐
                          │ Encoder x2  │──Down(64→128)  │
                          └─────────────┘                │
                                                   ┌───────▼───────┐
                                                   │ Middle x4 NAF │
                                                   │  blocks (128ch)│
                                                   └───────┬───────┘
                                                   Up(128→64) + skip
                                                   Decoder x2
                                          Up(64→32) + skip
                                          Decoder x2
                                                │
                                   PixelShuffle x2 SR head
                                          (32→128→PS(2)→32→1)
                                                │
                        Restored output (1x256x256, clamped [0,1])
```

- **Training strategy:** supervised end-to-end on paired GT/NoisyLR data,
  AdamW optimizer, cosine LR schedule, deterministic train/val split by
  filename (fixed seed) so validation never leaks into training.
- **Loss function design:** weighted sum of
  - Charbonnier loss (robust L1 — pixel fidelity without MSE's
    over-penalization of outliers)
  - SSIM loss (directly optimizes for a metric we're graded on)
  - Low-weight VGG perceptual loss (nudges LPIPS down without a GAN's
    hallucination risk — kept intentionally low-weight so it can't dominate)
- **Data augmentation:** geometric (random flip/rotation) plus, more
  importantly, *domain-randomized extra degradation* — additional
  randomized speckle/Gaussian noise layered on top of KLA's provided
  degradation during training, at varying strengths. This is aimed
  directly at the out-of-distribution generalization requirement: the
  model sees a distribution of noise strengths, not one fixed recipe, so
  it can't just memorize KLA's exact degradation parameters.

---

## Slide 5 — What Makes Our Approach Different

- **Joint denoise+SR in a single fused head**, rather than a
  denoise-then-upscale (or upscale-then-denoise) two-stage pipeline —
  avoids compounding errors across stages and keeps inference to one
  forward pass (helps both accuracy and speed).
- **Domain-randomized noise augmentation** specifically targeting the
  stated OOD generalization criterion, rather than only geometric
  augmentation (which most baseline pipelines stop at).
- **Deliberately GAN-free**: chose reliability/no-hallucination over the
  perceptual sharpness a GAN might add, directly because the problem
  statement calls out "artificial patterns or ringing" as a failure mode.
- **Unclipped-input training**: we noticed (and verified directly against
  the delivered dataset) that degraded input intensities exceed the [0,1]
  GT range — we deliberately do *not* clip this away before feeding the
  network, since those out-of-range pixels carry real information about
  where the noise hit hardest.
- **Empirically data-driven, not assumption-driven, tuning**: test-time
  augmentation (self-ensembling over flips/rotations) is a standard trick
  assumed to help — we actually measured it against our own checkpoint
  rather than assuming, and found it improves PSNR/SSIM slightly but
  makes LPIPS *measurably worse* (image-averaging smooths away exactly
  the fine texture LPIPS rewards preserving). We shipped with TTA off by
  default as a result — a concrete example of validating an assumption
  against real numbers before trusting it in the graded pipeline.
- **Capacity scaling was tested empirically, not guessed**: we ran short
  (5-10 epoch) sanity checks across five model configurations (0.76M up
  to 29.2M parameters) before committing GPU hours to any full 100-epoch
  run, comparing the actual marginal gain per size increase rather than
  assuming bigger is better.
- **Chose deployability over the single highest benchmark number**: our
  largest tested model (29.2M params, matching a published SEM-denoising
  benchmark's architecture) tied our shipped model's PSNR but needed a
  ~117MB checkpoint — over GitHub's 100MB plain-file limit, requiring Git
  LFS. Since the evaluation script must run unattended on KLA's grading
  hardware, and an LFS pointer-file failure there would silently zero the
  submission, we shipped the model that ties on top-line accuracy *and*
  is provably safe to deploy, instead of chasing a marginal gain at real
  operational risk.

---

## Slide 6 — Results

**On our held-out validation split** (320 images, deterministic seeded
split, never trained on):

| Metric | Score |
|---|---|
| **PSNR** | **28.83 dB** |
| **SSIM** | **0.7870** |
| **LPIPS** | **0.2237** |

*(`PROGRESS.md` in the repo has the full comparison table across every
model size we tried, including the 29.2M-param model that tied on PSNR
but wasn't shipped — see Slide 5.)*

- **Before / after / ground-truth comparison grid:** [FILL IN — render
  3-4 `.npy` triplets from `sample_outputs/` (restored) against the
  matching `train/NoisyLR/` (degraded input) and `train/GT/` (ground
  truth) files as PNGs for the slide. Pick one clean/easy case and one
  heavy-speckle case to show robustness across difficulty.]

---

## Slide 7 — Tech Stack & Performance

- **Framework:** PyTorch (2.x) + torchvision — pip-installable, no custom
  CUDA kernels or exotic dependencies.
- **Hardware used for training:** Kaggle's free-tier NVIDIA Tesla T4 GPU
  (chosen over Colab for its 9-hour background-session limit vs Colab's
  compute-unit throttling).
- **Training time:** ~6.5 hours for a full 100-epoch run on a T4.
- **Model size:** 6.65M parameters (~26.7 MB as float32 weights).
- **Inference time per image:** ~20 ms/image measured directly on a
  Kaggle T4 for our smaller width=64 configuration (single forward pass,
  no TTA); scaling by parameter count puts the shipped width=96 model at
  roughly ~40-50 ms/image on the same hardware — comfortably inside
  KLA's stated "10 seconds good / 10 minutes bad" bar by 2+ orders of
  magnitude. Expect faster still on KLA's H100 grading hardware.

---

## Slide 8 — Links

- **GitHub repository (mandatory):** https://github.com/Ganesh-0509/semicon
- **Demo video (optional but recommended, max 5 min):** [FILL IN]

---

## Slide 9 — References

- Chen, L., Chu, X., Zhang, X., Sun, J. — *"Simple Baselines for Image
  Restoration"* (NAFNet), ECCV 2022 — https://arxiv.org/abs/2204.04676
- Liang, J. et al. — *"SwinIR: Image Restoration Using Swin Transformer"*,
  ICCVW 2021 — https://arxiv.org/abs/2108.10257 (considered as an
  alternative; not used due to higher compute cost at inference)
- Zhang, R. et al. — *"The Unreasonable Effectiveness of Deep Features as
  a Perceptual Metric"* (LPIPS), CVPR 2018 — https://arxiv.org/abs/1801.03924
- Wang, Z. et al. — *"Image Quality Assessment: From Error Visibility to
  Structural Similarity"* (SSIM), IEEE TIP 2004
- KLA Problem Statement 1 — SEMICON India Hackathon 2026, dataset and
  problem brief provided by KLA
- Park, J., Oh, S., Jang, J. — *"Deep learning denoising enables rapid SEM
  imaging under charging conditions for FE SEM, CD SEM, and review SEM"*,
  Scientific Reports, 2025 — NAFNet applied to semiconductor SEM imagery;
  informed our capacity-scaling experiments

---

## Reminders before export
- Save as **PDF**, filename `TeamName_KLA_PS01.pdf`
- Max 8–9 slides, delete the instruction slide from the template
- Every `[FILL IN]` above must be replaced before submission — Slide 6 and
  7's numbers especially, since submitting placeholder metrics is worse
  than submitting late
