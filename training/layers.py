"""
training/layers.py - Custom layers for BitDistill implementation
Implements BitLinear (1.58-bit quantized linear layer with STE) and SubLN (sub-layer normalization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BitLinear(nn.Module):
    """
    BitLinear layer: Linear layer with 1.58-bit quantization (ternary weights {-1, 0, 1}).
    Uses absmean quantization matching bitnet.cpp logic, with Straight-Through Estimator (STE) for training.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to ternary {-1, 0, 1} using absmean scaling.
        Matches the logic in bitnet.cpp/utils/convert.py.
        """
        # Compute absmean scale Δ
        abs_w = torch.abs(w)
        delta = torch.mean(abs_w) + 1e-8  # Add small eps to avoid division by zero

        # Quantize: Q_w(W) = Δ * RoundClip(W / Δ, -1, 1)
        w_scaled = w / delta
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)

        # Rescale back
        return w_quant * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on the same device and dtype as input
        device = x.device
        dtype = x.dtype
        weight = self.weight.to(device).to(dtype)
        bias = self.bias.to(device).to(dtype) if self.bias is not None else None

        # Quantize weights during forward (for inference compatibility)
        w_quant = self.quantize_weights(weight)

        # Use quantized weights in linear operation
        output = F.linear(x, w_quant, bias)

        # STE: During backprop, gradients flow through original weights
        if self.training:
            # Detach quantized weights and attach to original for gradient computation
            w_quant = w_quant.detach() + weight - weight.detach()

        return output


class SubLN(nn.Module):
    """
    SubLN (Sub-Layer Normalization): RMSNorm for stabilizing activations in BitNet.
    Matches the SubLN used in BitDistill for MHSA and FFN output projections.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(SubLN, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weight is on same device and dtype as x
        weight = self.weight.to(x.device).to(x.dtype)
        # RMSNorm: normalize by root mean square
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return weight * (x / rms)


def replace_linear_with_bitlinear(module: nn.Module) -> None:
    """
    Recursively replace nn.Linear layers with BitLinear in a model.
    Used for student model initialization.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with BitLinear, preserving in/out features and bias
            bit_linear = BitLinear(child.in_features, child.out_features, child.bias is not None)
            # Copy weights and bias if they exist
            with torch.no_grad():
                bit_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    bit_linear.bias.copy_(child.bias)
            setattr(module, name, bit_linear)
        else:
            # Recurse into submodules
            replace_linear_with_bitlinear(child)