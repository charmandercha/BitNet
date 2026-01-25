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
    BitLinear layer: Linear layer with 1.58-bit quantization (ternary weights {-1, 0, 1} and 8-bit activations).
    Uses absmean quantization for weights and absmax for activations, with Straight-Through Estimator (STE) for training.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.eps = 1e-5

    def quantize_weights(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CORRECTED: Quantize weights to ternary {-1, 0, 1} using absmean scaling.
        Formula: Q_w(W) = Δ * RoundClip(W/Δ, -1, 1) where Δ = mean(|W|)
        """
        # Compute absmean scale Δ as per paper Formula 1
        gamma = torch.mean(torch.abs(w)) + self.eps
        
        # Quantize: RoundClip(W/Δ, -1, 1)
        w_scaled = w / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        
        # Rescale back: Δ * quantized_value
        w_quant = w_quant * gamma
        
        return w_quant, gamma

    def quantize_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CORRECTED: Quantize activations to 8-bits [-128, 127] using absmax scaling.
        Formula: Q_INT8(X) = (γ/127) * RoundClip(127*X/(γ+ε), -128, 127) where γ = max(|X|)
        """
        # Compute absmax scale γ as per paper Formula 2
        gamma = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + self.eps
        
        # Scale to 8-bit range: 127*X/γ
        x_scaled = 127.0 * x / gamma
        
        # RoundClip to [-128, 127] range
        x_quant = torch.clamp(torch.round(x_scaled), -128, 127)
        
        # Rescale back: (γ/127) * quantized_value
        x_quant = gamma * x_quant / 127.0
        
        return x_quant, gamma

def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on same device and dtype as input
        device = x.device
        dtype = x.dtype
        weight = self.weight.to(device).to(dtype)
        bias = self.bias.to(device).to(dtype) if self.bias is not None else None

        # Quantize weights and activations
        w_quant, gamma = self.quantize_weights(weight)
        x_quant, scale = self.quantize_activations(x)

        # STE for training - CORRECTED IMPLEMENTATION
        # PyTorch automatically handles STE for clamp/round operations
        if self.training:
            # Use quantized values for forward pass
            w_final = w_quant * gamma
            x_final = x_quant / scale
            # Add residual connection for STE gradient flow
            residual_w = weight - weight.detach()  # Gradients flow through original weights
            residual_x = x - x.detach()          # Gradients flow through original input
            output = F.linear(x_final, w_final, bias) + F.linear(residual_x, residual_w, None)
        else:
            w_final = w_quant * gamma
            x_final = x_quant / scale
            output = F.linear(x_final, w_final, bias)

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