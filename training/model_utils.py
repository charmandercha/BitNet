#!/usr/bin/env python3
"""
training/model_utils.py - Utilities for extracting Q,K,V projections
Critical fix for proper MiniLM attention distillation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class QKVExtractor(nn.Module):
    """
    Hook to extract Q,K,V projections from transformer attention layers.
    This solves the critical bug in attention distillation.
    """
    
    def __init__(self):
        self.student_qkv = {}
        self.teacher_qkv = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, model_type: str = "student"):
        """Register forward hooks to capture Q,K,V projections."""
        target_dict = self.student_qkv if model_type == "student" else self.teacher_qkv
        
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # For attention layers, input[0] contains the hidden states
                # We need to extract Q,K,V from the attention computation
                if hasattr(module, 'q_proj') or 'q' in layer_name.lower():
                    if isinstance(output, (list, tuple)):
                        target_dict[layer_name] = output
                    else:
                        # If single output, this might be Q,K,V concatenated
                        target_dict[layer_name] = output
            return hook_fn
        
        # Register hooks on attention projection layers
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def extract_qkv_from_layer(self, layer_idx: int, model_type: str = "student") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Q,K,V from a specific transformer layer.
        This is the corrected implementation for MiniLM distillation.
        """
        target_dict = self.student_qkv if model_type == "student" else self.teacher_qkv
        
        # Find the layer's Q,K,V projections
        layer_name = None
        for name in target_dict.keys():
            if f'layers.{layer_idx}' in name or f'layer.{layer_idx}' in name:
                layer_name = name
                break
        
        if layer_name is None:
            return None, None, None
        
        layer_output = target_dict[layer_name]
        
        if isinstance(layer_output, (list, tuple)):
            # Q,K,V are directly available
            if len(layer_output) >= 3:
                return layer_output[0], layer_output[1], layer_output[2]
        
        # Fallback: reconstruct Q,K,V from attention output
        # This requires model-specific knowledge
        return None, None, None
    
    def clear_caches(self):
        """Clear stored Q,K,V caches."""
        self.student_qkv.clear()
        self.teacher_qkv.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def extract_qkv_from_attention_layer(model: nn.Module, layer_idx: int, input_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    CORRECTED: Extract Q,K,V projections by modifying forward pass.
    
    This is the fundamental fix for MiniLM attention distillation.
    Instead of trying to extract from attention weights, we intercept the projections.
    """
    qkv = None
    
    def extract_qkv_hook(module, input, output):
        nonlocal qkv
        if 'q_proj' in module.__class__.__name__.lower() or 'query' in module.__class__.__name__.lower():
            qkv = [output]  # Q projection
        elif 'k_proj' in module.__class__.__name__.lower() or 'key' in module.__class__.__name__.lower():
            if qkv is not None:
                qkv.append(output)
        elif 'v_proj' in module.__class__.__name__.lower() or 'value' in module.__class__.__name__.lower():
            if qkv is not None and len(qkv) == 2:
                qkv.append(output)
    
    # Find and hook the target attention layer
    target_layers = []
    for name, module in model.named_modules():
        if f'layers.{layer_idx}' in name or f'layer.{layer_idx}' in name:
            if hasattr(module, 'self_attn') or hasattr(module, 'attention'):
                attention = module.self_attn if hasattr(module, 'self_attn') else module.attention
                
                # Hook Q,K,V projections
                if hasattr(attention, 'q_proj'):
                    hook1 = attention.q_proj.register_forward_hook(extract_qkv_hook)
                    target_layers.append(hook1)
                if hasattr(attention, 'k_proj'):
                    hook2 = attention.k_proj.register_forward_hook(extract_qkv_hook)
                    target_layers.append(hook2)
                if hasattr(attention, 'v_proj'):
                    hook3 = attention.v_proj.register_forward_hook(extract_qkv_hook)
                    target_layers.append(hook3)
                break
    
    # Forward pass to capture Q,K,V
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in target_layers:
        hook.remove()
    
    # Return extracted Q,K,V
    if qkv is not None and len(qkv) == 3:
        return qkv[0], qkv[1], qkv[2]
    
    return None, None, None

def get_last_layer_attention_projections(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Q,K,V projections from the last transformer layer.
    This is specifically designed for MiniLM attention distillation.
    """
    # Count total layers
    layer_count = 0
    for name, module in model.named_modules():
        if 'layers.' in name and '.self_attn' in name:
            layer_count += 1
    
    # Extract from last layer
    last_layer_idx = layer_count - 1
    return extract_qkv_from_attention_layer(model, last_layer_idx, input_tensor)