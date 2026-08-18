"""
NAFNet-lite: a compact NAFNet-style encoder-decoder (Chen et al., 2022,
"Simple Baselines for Image Restoration") adapted for this task:
  - single-channel (grayscale) in/out
  - input at 128x128, output at 256x256 -> a PixelShuffle x2 head is
    fused onto the end of the restoration decoder, so denoising and
    super-resolution happen in one pass instead of two separate stages.

Chosen over SwinIR/Restormer for this hackathon because NAFNet gets
competitive-to-SOTA denoising/deblurring quality at much lower compute
(no self-attention, no nonlinear activations) -- and the KLA benchmark
explicitly penalizes slow inference on the H100 grading run.

No adversarial (GAN) training is used anywhere in this project: the
problem statement explicitly warns against "artificial patterns or
ringing", which is the classic GAN-restoration failure mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """One NAFNet block: simplified channel attention + SimpleGate,
    two residual sub-blocks, learnable per-channel residual scales."""

    def __init__(self, channels, expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma

        return x


class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.op(x)


class NAFNetLiteSR(nn.Module):
    """
    width: base channel count (32 keeps this fast on H100; bump to 48/64
           if there's inference-time headroom left after benchmarking).
    enc_blocks / dec_blocks: block counts per encoder/decoder stage.
    middle_blocks: block count at the bottleneck.
    """

    def __init__(self, in_ch=1, out_ch=1, width=32, enc_blocks=(2, 2), middle_blocks=4, dec_blocks=(2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(Down(ch))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(Up(ch))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        # Fused 2x super-resolution head: width -> 4*width -> PixelShuffle(2) -> width, at 2x spatial res.
        self.sr_expand = nn.Conv2d(width, width * 4, 3, padding=1)
        self.sr_shuffle = nn.PixelShuffle(2)
        self.sr_refine = NAFBlock(width)
        self.out_conv = nn.Conv2d(width, out_ch, 3, padding=1)

        self.padder_size = 2 ** len(enc_blocks)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)

    def forward(self, x):
        x, (orig_h, orig_w) = self._pad_to_multiple(x)

        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.sr_expand(x)
        x = self.sr_shuffle(x)
        x = self.sr_refine(x)
        x = self.out_conv(x)

        # Crop back from any reflect-padding, then account for the 2x SR head.
        x = x[:, :, : orig_h * 2, : orig_w * 2]
        return x


def build_model(width=32, enc_blocks=(2, 2), middle_blocks=4, dec_blocks=(2, 2)):
    return NAFNetLiteSR(in_ch=1, out_ch=1, width=width, enc_blocks=enc_blocks,
                         middle_blocks=middle_blocks, dec_blocks=dec_blocks)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(1, 1, 128, 128)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}, params={n_params/1e6:.2f}M")
