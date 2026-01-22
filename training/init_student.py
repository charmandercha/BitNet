"""
training/init_student.py - Initialize BitNet student model with BitLinear and SubLN
Loads a small HF model config, replaces Linear layers with BitLinear, and adds SubLN as per BitDistill.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from .layers import BitLinear, SubLN, replace_linear_with_bitlinear
from typing import Optional

def apply_subLN_to_model(model: nn.Module) -> None:
    """
    Apply SubLN modifications to the model architecture as described in BitDistill Stage-1.
    Adds SubLN before MHSA output projection and FFN down projection.
    Assumes Llama/Qwen-like architecture.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'o_proj'):
            # Add SubLN before MHSA output projection
            original_o_proj = module.self_attn.o_proj
            subln = SubLN(original_o_proj.in_features)
            module.self_attn.o_proj = nn.Sequential(subln, original_o_proj)
        elif hasattr(module, 'mlp') and hasattr(module.mlp, 'down_proj'):
            # Add SubLN before FFN down projection
            original_down_proj = module.mlp.down_proj
            subln = SubLN(original_down_proj.in_features)
            module.mlp.down_proj = nn.Sequential(subln, original_down_proj)

def init_bitnet_student(model_name: str = "/home/marcos/BitNet/HY-MT1.5-1.8B", device: str = "cuda") -> nn.Module:
    """
    Initialize a BitNet student model:
    1. Load HF config and model
    2. Replace Linear layers with BitLinear
    3. Apply SubLN modifications
    """
    print(f"Loading model config: {model_name}")

    # Load config and model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
    model.to(device)

    print("Replacing Linear layers with BitLinear...")
    replace_linear_with_bitlinear(model)

    print("Applying SubLN modifications...")
    apply_subLN_to_model(model)

    print("BitNet student model initialized!")
    return model

if __name__ == "__main__":
    # Example usage
    student = init_bitnet_student()
    print(f"Model type: {type(student)}")
    print(f"Sample layer: {next(student.named_modules())}")

    # Quick test forward pass
    dummy_input = torch.randint(0, 1000, (1, 10)).to("cuda")  # Dummy token IDs
    with torch.no_grad():
        output = student(dummy_input)
    print(f"Forward pass successful, output shape: {output.logits.shape}")