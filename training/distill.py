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

def compute_attention_distillation_loss(student_attentions, teacher_attentions, distill_layer: int = -1):
    """
    Compute attention distillation using MSE on last layer to avoid OOM.
    """
    s_attn = student_attentions[distill_layer]
    t_attn = teacher_attentions[distill_layer]
    ad_loss = F.mse_loss(s_attn, t_attn)
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
    scaler = GradScaler('cuda')

    num_epochs = 3
    accumulation_steps = 8
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    teacher_outputs = teacher(**batch, output_attentions=True)

                student_outputs = student(**batch, output_attentions=True)

                ce_loss = nn.functional.cross_entropy(
                    student_outputs.logits[..., :-1, :].contiguous().view(-1, student_outputs.logits.size(-1)),
                    batch["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=-100
                )

                distill_loss, ld_loss, ad_loss = distillation_loss(student_outputs, teacher_outputs)
                total_loss = ce_loss + distill_loss
                total_loss = total_loss / accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                if step % 10 == 0:
                    print(f"Step {step}: Total Loss={total_loss.item() * accumulation_steps:.4f}, CE={ce_loss.item():.4f}, LD={ld_loss.item():.4f}, AD={ad_loss.item():.4f}, Grad Norm={grad_norm:.4f}")
    
    # Save student
    student.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    print("Stage-3 Distillation complete. Final student saved.")

if __name__ == "__main__":
    main()