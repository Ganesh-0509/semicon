"""
Loss functions for training.

Deliberately no adversarial/GAN loss anywhere -- the problem statement
explicitly warns against "artificial patterns or ringing", which is the
classic failure mode of GAN-based restoration (Real-ESRGAN etc.).

Combined loss = Charbonnier (robust pixel fidelity)
              + SSIM         (structural similarity, matches a grading metric)
              + small-weight VGG perceptual (helps the LPIPS grading metric,
                kept low-weight so it can't dominate and hallucinate detail)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Robust L1 (smooth near zero) -- standard for restoration tasks,
    less prone to over-smoothing than plain L2/MSE."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d[None, None, :, :]


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window. Single-channel only
    (matches the grayscale-only nature of this dataset)."""

    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.register_buffer("_window_cache", torch.empty(0), persistent=False)

    def _ssim_map(self, pred, target):
        window = _gaussian_window(self.window_size, self.sigma, pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)

        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
            (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
        )
        return ssim_map

    def forward(self, pred, target):
        return 1.0 - self._ssim_map(pred, target).mean()


class VGGPerceptualLoss(nn.Module):
    """Optional. Replicates the single grayscale channel to 3 channels
    to run through an ImageNet-pretrained VGG16. Requires internet on
    first use (downloads pretrained weights) -- fine on a cloud GPU
    training environment, not needed at KLA's inference/eval time."""

    def __init__(self, layer_idx=16):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        f_pred = self.vgg(self._prep(pred))
        f_target = self.vgg(self._prep(target))
        return F.l1_loss(f_pred, f_target)


class CombinedLoss(nn.Module):
    def __init__(self, w_charbonnier=1.0, w_ssim=0.2, w_perceptual=0.05, use_perceptual=True):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual

        self.perceptual = None
        if use_perceptual:
            try:
                self.perceptual = VGGPerceptualLoss()
            except Exception as e:
                print(f"[losses] Perceptual loss disabled (couldn't load VGG weights: {e})")
                self.perceptual = None

    def forward(self, pred, target):
        pred_c = pred.clamp(0, 1)
        target_c = target.clamp(0, 1)

        loss_c = self.charbonnier(pred, target)
        loss_s = self.ssim(pred_c, target_c)
        total = self.w_charbonnier * loss_c + self.w_ssim * loss_s

        logs = {"charbonnier": loss_c.item(), "ssim_loss": loss_s.item()}

        if self.perceptual is not None:
            loss_p = self.perceptual(pred_c, target_c)
            total = total + self.w_perceptual * loss_p
            logs["perceptual"] = loss_p.item()

        logs["total"] = total.item()
        return total, logs
