"""
training/distill.py - Refactored Dual-Signal Distillation Engine for BitDistill
Updated for PyTorch 2.x AMP API, device synchronization, and gradient monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from .init_student import init_bitnet_student
from typing import Dict, Any

def load_teacher_model(model_path: str, device: str = "cuda") -> nn.Module:
    """Load teacher model in eval mode without gradients."""
    print(f"Loading teacher model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_student_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Load student model from Stage-2 checkpoint."""
    print(f"Loading student model from {checkpoint_path}")
    student = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    student.to(device)
    return student

def compute_attention_distillation_loss(student_attentions, teacher_attentions, distill_layer: int = -1, split_heads: int = 1):
    """
    Compute multi-head attention distillation loss as per BitDistill Algorithm 1.
    distills from the last layer by default.
    """
    # student_attentions: list of tensors [batch, num_heads, seq_len, seq_len]
    # Assume distill_layer is the index, -1 for last
    s_attn = student_attentions[distill_layer]  # [B, H, L, L]
    t_attn = teacher_attentions[distill_layer]  # [B, H, L, L]

    # Flatten heads if split_heads > 1, but for simplicity, assume split_heads=1
    B, H, L, _ = s_attn.shape
    s_attn_flat = s_attn.view(B * H, L, L)  # [B*H, L, L]
    t_attn_flat = t_attn.view(B * H, L, L)

    # Compute KL on each [L, L] matrix
    s_log = torch.log(s_attn_flat + 1e-8)
    t_prob = t_attn_flat + 1e-8
    ad_loss = F.kl_div(s_log, t_prob, reduction="batchmean")
    return ad_loss

def distillation_loss(student_outputs, teacher_outputs, temperature: float = 2.0, lambda_ld: float = 1.0, gamma_ad: float = 0.001):
    """
    Compute BitDistill distillation loss:
    - Logits Distillation (LD): KL on softened logits
    - Multi-Head Attention Distillation (AD): KL on attention matrices
    """
    device = student_outputs.logits.device
    teacher_outputs.logits = teacher_outputs.logits.to(device)
    teacher_outputs.attentions = [a.to(device) for a in teacher_outputs.attentions]

    # Logits Distillation (Reverse KL as per MiniLLM for better generation)
    student_logits = student_outputs.logits / temperature
    teacher_logits = teacher_outputs.logits / temperature
    ld_loss = F.kl_div(
        F.log_softmax(teacher_logits, dim=-1),
        F.softmax(student_logits, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    # Attention Distillation (from last layer)
    ad_loss = compute_attention_distillation_loss(student_outputs.attentions, teacher_outputs.attentions)

    # Total distillation loss
    total_distill_loss = lambda_ld * ld_loss + gamma_ad * ad_loss
    return total_distill_loss, ld_loss, ad_loss

def prepare_dataset(tokenizer, max_length: int = 512):
    """Load and prepare WikiText-103 dataset for language modeling."""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return tokenized_dataset, data_collator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/marcos/BitNet/HY-MT1.5-1.8B"
    student_checkpoint = "/home/marcos/BitNet/student_stage2_checkpoints"
    
    # Load teacher and student (after Stage-2)
    teacher = load_teacher_model(model_path, device)
    student = load_student_model(student_checkpoint, device)
    student.train()
    student.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset
    train_dataset, data_collator = prepare_dataset(tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)
    scaler = GradScaler('cuda')  # Updated for torch.amp
    
    num_epochs = 3
    step = 0
    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Ensure batch on device
            
            with autocast('cuda', dtype=torch.float16):  # Updated for torch.amp
                # Forward teacher (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher(**batch, output_attentions=True)
                
                # Forward student
                student_outputs = student(**batch, output_attentions=True)
                
                # Compute CE loss
                ce_loss = nn.functional.cross_entropy(
                    student_outputs.logits[..., :-1, :].contiguous().view(-1, student_outputs.logits.size(-1)),
                    batch["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=-100
                )
                
                # Compute distillation loss
                distill_loss, ld_loss, ad_loss = distillation_loss(student_outputs, teacher_outputs)
                
                # Total loss
                total_loss = ce_loss + distill_loss
            
            # Backward
            scaler.scale(total_loss).backward()
            
            # Gradient monitoring and clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            if grad_norm == 0:
                print(f"Warning: Zero gradient norm at step {step}. Check STE in BitLinear.")
            elif grad_norm > 10:
                print(f"Warning: Exploding gradients (norm={grad_norm:.4f}) at step {step}.")
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}: Total Loss={total_loss.item():.4f}, CE={ce_loss.item():.4f}, LD={ld_loss.item():.4f}, AD={ad_loss.item():.4f}, Grad Norm={grad_norm:.4f}")
    
    # Save student
    student.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    print("Stage-3 Distillation complete. Final student saved.")

if __name__ == "__main__":
    main()