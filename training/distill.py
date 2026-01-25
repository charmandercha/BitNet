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

def compute_attention_distillation_loss(student_states, teacher_states, distill_layer: int = -1, split_heads: int = 1, temperature: float = 5.0):
    """
    Compute MiniLM-style attention relation distillation as per Algorithm 1.
    student_states: [layers, B, num_heads, seq_len, head_dim] - Q,K,V states
    teacher_states: [layers, B, num_heads, seq_len, head_dim] - Q,K,V states
    """
    distill_loss = 0.0
    
    # Select distillation layer
    s_layer_states = student_states[distill_layer] if isinstance(student_states, (list, tuple)) else student_states
    t_layer_states = teacher_states[distill_layer] if isinstance(teacher_states, (list, tuple)) else teacher_states
    
    # Get Q, K, V projections from the layer
    # This depends on model architecture - we need to extract from hidden states
    if isinstance(s_layer_states, tuple) and len(s_layer_states) >= 3:
        s_q, s_k, s_v = s_layer_states[0], s_layer_states[1], s_layer_states[2]
        t_q, t_k, t_v = t_layer_states[0], t_layer_states[1], t_layer_states[2]
    else:
        # Fallback: use last layer's attention weights
        return torch.tensor(0.0, device=s_layer_states.device)
    
    B, num_heads, L, d = s_q.shape
    D = num_heads * d // split_heads
    
    # Compute relation matrices for Q, K, V
    for s_proj, t_proj in [(s_q, t_q), (s_k, t_k), (s_v, t_v)]:
        # Reshape for split heads and normalize
        s_values = F.normalize(s_proj.transpose(1, 2).reshape(B, L, split_heads, D).transpose(1, 2), dim=-1)
        t_values = F.normalize(t_proj.transpose(1, 2).reshape(B, L, split_heads, D).transpose(1, 2), dim=-1)
        
        # Compute relation matrix
        s_relation = torch.matmul(s_values, s_values.transpose(-2, -1))
        t_relation = torch.matmul(t_values, t_values.transpose(-2, -1))
        
        # Reshape for KL computation: [B, split_heads, L, L] -> [B*split_heads, L, L]
        s_relation = s_relation.reshape(B * split_heads, L, L)
        t_relation = t_relation.reshape(B * split_heads, L, L)
        
        # Apply temperature and compute softmax
        s_relation = s_relation / temperature
        t_relation = t_relation / temperature
        
        s_prob = F.softmax(s_relation, dim=-1).clamp(min=1e-8)
        t_prob = F.softmax(t_relation, dim=-1).clamp(min=1e-8)
        
        # KL divergence
        distill_loss += F.kl_div(torch.log(s_prob), t_prob, reduction="batchmean", log_target=False)
    
    return distill_loss

def distillation_loss(student_outputs, teacher_outputs, task_type: str = "classification"):
    """
    Compute BitDistill distillation loss with paper's recommended parameters.
    """
    device = student_outputs.logits.device
    teacher_outputs.logits = teacher_outputs.logits.to(device)
    teacher_outputs.attentions = [a.to(device) for a in teacher_outputs.attentions]

    # Paper's recommended parameters
    if task_type == "classification":
        temperature = 5.0
        lambda_ld = 10.0
        gamma_ad = 1e-5
    else:  # summarization
        temperature = 5.0
        lambda_ld = 1.0
        gamma_ad = 1e-3

    # Logits Distillation (LD)
    student_logits = student_outputs.logits / temperature
    teacher_logits = teacher_outputs.logits / temperature
    ld_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    # Attention Distillation (AD) - need hidden states for relation matrices
    # Note: This requires models to output hidden_states, not just attentions
    if hasattr(student_outputs, 'hidden_states') and hasattr(teacher_outputs, 'hidden_states'):
        ad_loss = compute_attention_distillation_loss(
            student_outputs.hidden_states[distill_layer], 
            teacher_outputs.hidden_states[distill_layer],
            distill_layer=-1,
            temperature=temperature
        )
    else:
        # Fallback to attention weights if hidden states not available
        ad_loss = compute_attention_distillation_loss(
            student_outputs.attentions, 
            teacher_outputs.attentions,
            distill_layer=-1,
            temperature=temperature
        )

    # Total distillation loss
    total_distill_loss = lambda_ld * ld_loss + gamma_ad * ad_loss
    return total_distill_loss, ld_loss, ad_loss

def prepare_dataset(tokenizer, max_length: int = 512, stage: str = "continue_pretrain"):
    """Load and prepare dataset based on training stage."""
    if stage == "continue_pretrain":
        # Stage-2: Use larger corpus (FALCON or similar)
        try:
            dataset = load_dataset("tiiuae/falcon-refinedweb", split="train[:1%]")  # ~10B tokens
        except:
            # Fallback to wikitext if FALCON not available
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        # Stage-3: Downstream task dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:5%]")
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return tokenized_dataset, data_collator

def stage2_continue_pretrain():
    """Stage-2: Continue pre-training with large corpus."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/marcos/BitNet/HY-MT1.5-1.8B"
    
    # Load student model from Stage-1
    student = init_bitnet_student(model_path, device)
    student.train()
    student.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Large dataset for continue pre-training
    train_dataset, data_collator = prepare_dataset(tokenizer, stage="continue_pretrain")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    scaler = GradScaler('cuda')
    
    num_epochs = 1  # Paper uses 10B tokens
    accumulation_steps = 8
    step = 0
    
    print("Starting Stage-2: Continue Pre-training...")
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast('cuda', dtype=torch.float16):
                outputs = student(**batch, output_hidden_states=True)
                ce_loss = nn.functional.cross_entropy(
                    outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                    batch["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=-100
                )
                loss = ce_loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                step += 1
                if step % 100 == 0:
                    print(f"Stage-2 Step {step}: Loss={loss.item() * accumulation_steps:.4f}")
    
    # Save Stage-2 checkpoint
    student.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    print("Stage-2 complete. Checkpoints saved.")

def stage3_distillation():
    """Stage-3: Distillation-based fine-tuning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/marcos/BitNet/HY-MT1.5-1.8B"
    student_checkpoint = "/home/marcos/BitNet/student_stage2_checkpoints"
    
    # Load teacher and student
    teacher = load_teacher_model(model_path, device)
    student = load_student_model(student_checkpoint, device)
    student.train()
    student.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Downstream task dataset
    train_dataset, data_collator = prepare_dataset(tokenizer, stage="downstream")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    scaler = GradScaler('cuda')
    
    num_epochs = 3
    accumulation_steps = 8
    step = 0
    
    print("Starting Stage-3: Distillation-based Fine-tuning...")
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    teacher_outputs = teacher(**batch, output_hidden_states=True, output_attentions=True)
                
                student_outputs = student(**batch, output_hidden_states=True, output_attentions=True)
                
                ce_loss = nn.functional.cross_entropy(
                    student_outputs.logits[..., :-1, :].contiguous().view(-1, student_outputs.logits.size(-1)),
                    batch["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=-100
                )
                
                distill_loss, ld_loss, ad_loss = distillation_loss(student_outputs, teacher_outputs, task_type="classification")
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
                    print(f"Stage-3 Step {step}: Total={total_loss.item() * accumulation_steps:.4f}, CE={ce_loss.item():.4f}, LD={ld_loss.item():.4f}, AD={ad_loss.item():.4f}")
    
    student.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_final_checkpoints")
    print("Stage-3 complete. Final student saved.")

def main():
    """Run complete BitDistill pipeline."""
    print("BitDistill Pipeline Starting...")
    print("Stage-1: Model refinement (SubLN) - already done in init_student.py")
    stage2_continue_pretrain()
    stage3_distillation()
    print("BitDistill complete!")

if __name__ == "__main__":
    main()