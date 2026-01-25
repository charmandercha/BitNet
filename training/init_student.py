"""
training/init_student.py - Initialize BitNet student model with BitLinear and SubLN
Loads a small HF model config, replaces Linear layers with BitLinear, and adds SubLN as per BitDistill.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from .layers import BitLinear, SubLN, replace_linear_with_bitlinear, BitNetAttentionWithSubLN, BitNetFFNWithSubLN
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
    
    def replace_recursive(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, nn.Linear):
                if full_name in teacher_modules:
                    bit_layer = BitLinear(child.in_features, child.out_features, child.bias is not None)
                    calibrate_bitlinear_weights(bit_layer, teacher_modules[full_name])
                    setattr(module, name, bit_layer)
            else:
                replace_recursive(child, full_name)
    
    replace_recursive(model)

def apply_subLN_to_model(model: nn.Module) -> None:
    """
    CORRECTED: Apply SubLN modifications as per BitDistill paper Equations 4-5.
    Replace entire attention and FFN modules with custom implementations that integrate SubLN.
    
    Equation 4: Yl = Xl + SubLN(Concat(heads)) * Wout^MHSA
    Equation 5: Xl+1 = Yl + SubLN(Yl * Wup) ⊙ σ(Yl * Wgate) * Wdown^FFN
    """
    def apply_recursive(module):
        for name, child in module.named_children():
            # Check for attention modules - replace entire attention with SubLN version
            if hasattr(child, 'self_attn'):
                # Replace entire attention module with SubLN-integrated version
                child.self_attn = BitNetAttentionWithSubLN(child.self_attn)
            elif hasattr(child, 'attention'):
                child.attention = BitNetAttentionWithSubLN(child.attention)
            
            # Check for FFN modules - replace entire FFN with SubLN version
            if hasattr(child, 'mlp'):
                child.mlp = BitNetFFNWithSubLN(child.mlp)
            elif hasattr(child, 'feed_forward'):
                child.feed_forward = BitNetFFNWithSubLN(child.feed_forward)
            
            # Recursively apply to children
            apply_recursive(child)
    
    apply_recursive(model)

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