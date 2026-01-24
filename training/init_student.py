"""
training/init_student.py - Initialize BitNet student model with BitLinear and SubLN
Loads a small HF model config, replaces Linear layers with BitLinear, and adds SubLN as per BitDistill.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from .layers import BitLinear, SubLN, replace_linear_with_bitlinear
from typing import Optional

def calibrate_bitlinear_weights(student_layer: BitLinear, teacher_layer: nn.Linear) -> None:
    """
    Calibrate BitLinear weights from teacher to avoid shock in warm-start.
    """
    with torch.no_grad():
        source_weight = teacher_layer.weight.data.clone()
        gamma = torch.mean(torch.abs(source_weight)) + 1e-8
        student_layer.weight.copy_(source_weight)
        if teacher_layer.bias is not None and student_layer.bias is not None:
            student_layer.bias.copy_(teacher_layer.bias.data)

def apply_bitnet_architecture(model: nn.Module, teacher_model: nn.Module) -> None:
    """
    Replace Linear with BitLinear and calibrate weights.
    """
    teacher_modules = dict(teacher_model.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name in teacher_modules:
                bit_layer = BitLinear(module.in_features, module.out_features, module.bias is not None)
                calibrate_bitlinear_weights(bit_layer, teacher_modules[name])
                parent_name = ".".join(name.split(".")[:-1])
                layer_name = name.split(".")[-1]
                if parent_name:
                    setattr(dict(model.named_modules())[parent_name], layer_name, bit_layer)
                else:
                    setattr(model, layer_name, bit_layer)

def apply_subLN_to_model(model: nn.Module) -> None:
    """
    Apply SubLN modifications. Make robust by checking common names.
    """
    for name, module in model.named_modules():
        # Check for MHSA output projection
        if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'o_proj'):
            original_o_proj = module.self_attn.o_proj
            subln = SubLN(original_o_proj.in_features)
            module.self_attn.o_proj = nn.Sequential(subln, original_o_proj)
        elif hasattr(module, 'attention') and hasattr(module.attention, 'wo'):
            # Alternative naming
            original_o_proj = module.attention.wo
            subln = SubLN(original_o_proj.in_features)
            module.attention.wo = nn.Sequential(subln, original_o_proj)
        # Check for FFN down projection
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'down_proj'):
            original_down_proj = module.mlp.down_proj
            subln = SubLN(original_down_proj.in_features)
            module.mlp.down_proj = nn.Sequential(subln, original_down_proj)
        elif hasattr(module, 'feed_forward') and hasattr(module.feed_forward, 'w2'):
            # Alternative naming
            original_down_proj = module.feed_forward.w2
            subln = SubLN(original_down_proj.in_features)
            module.feed_forward.w2 = nn.Sequential(subln, original_down_proj)

def init_bitnet_student(model_path: str = "/home/marcos/BitNet/HY-MT1.5-1.8B", device: str = "cuda") -> nn.Module:
    """
    Initialize BitNet student with warm-start calibration.
    """
    print(f"Loading teacher from {model_path}")
    teacher = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

    config = AutoConfig.from_pretrained(model_path)
    student = AutoModelForCausalLM.from_config(config).to(device).to(torch.float16)

    print("Applying BitNet architecture with calibration...")
    apply_bitnet_architecture(student, teacher)

    print("Applying SubLN...")
    apply_subLN_to_model(student)

    del teacher
    torch.cuda.empty_cache()

    print("BitNet student initialized!")
    return student

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