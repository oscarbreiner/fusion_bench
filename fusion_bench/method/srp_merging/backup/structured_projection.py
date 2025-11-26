# structured_projection.py
from __future__ import annotations
from typing import Literal
import math
import torch
from torch import Tensor

TransformType = Literal["fwht", "srht", "dct", "dht"]

def next_pow2(n: int) -> int:
    """Next power of two >= n."""
    return 1 << (n - 1).bit_length()

@torch.no_grad()
def fwht_inplace_ortho(x: Tensor) -> Tensor:
    """
    In-place orthonormal Fast Walsh–Hadamard Transform (FWHT) on last dim.
    Requires last dimension to be a power of 2. Scaled by 1/sqrt(n).
    """
    n = x.shape[-1]
    if n <= 1:
        return x
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.view(-1, n)
        h *= 2
    x.mul_(1.0 / math.sqrt(n))
    return x

@torch.no_grad()
def dct_ortho(x: Tensor) -> Tensor:
    """Orthonormal DCT-II along last dim."""
    return torch.fft.dct(x, type=2, norm="ortho")

@torch.no_grad()
def idct_ortho(x: Tensor) -> Tensor:
    """Inverse (DCT-III) for DCT-II under orthonormal scaling."""
    return torch.fft.idct(x, type=2, norm="ortho")

@torch.no_grad()
def dht_ortho(x: Tensor) -> Tensor:
    """
    Orthonormal Discrete Hartley Transform along last dim.
    Using FFT: H(x)[k] = Re(FFT(x))[k] - Im(FFT(x))[k] with unitary FFT.
    Self-inverse under orthonormal normalization.
    """
    X = torch.fft.fft(x, norm="ortho")
    return (X.real - X.imag)

# DHT is self-inverse with orthonormal scaling
idht_ortho = dht_ortho

class OrthoTransform:
    """
    Wrap an orthonormal transform F with:
      - forward: F(x)
      - inverse: F^{-1}(x) == F^T(x)
      - length L (may be padded vs original D for Hadamard)
      - pad policy for hadamard (power-of-2 only)
    """
    def __init__(self, kind: TransformType, D: int, device: torch.device):
        self.kind = kind
        if kind in ("fwht", "srht"):
            self.L = next_pow2(D)   # Hadamard requires power-of-2
            self.F = fwht_inplace_ortho
            self.Finv = fwht_inplace_ortho  # orthonormal => self-inverse
            self.needs_pad = (self.L != D)
        elif kind == "dct":
            self.L = D
            self.F = dct_ortho
            self.Finv = idct_ortho
            self.needs_pad = False
        elif kind == "dht":
            self.L = D
            self.F = dht_ortho
            self.Finv = idht_ortho
            self.needs_pad = False
        else:
            raise ValueError(f"Unknown transform kind: {kind}")
        self.device = device

    def pad(self, x: Tensor, D: int) -> Tensor:
        if self.needs_pad and D < self.L:
            return torch.nn.functional.pad(x, (0, self.L - D))
        return x
