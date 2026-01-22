"""
training/continue_pretrain.py - Stage-2: Continued Pre-Training for BitDistill
Adapt the BitNet student to ternary weights using a small pretraining corpus.
Follows BitDistill Stage-2 to mitigate scalability issues.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from .init_student import init_bitnet_student
import math

def prepare_corpus(tokenizer, max_length: int = 512, num_samples: int = 10000):
    """Load and prepare a small pretraining corpus (subset of wikitext for demo, scale to FALCON in production)."""
    # For demo, use wikitext-103 subset; in practice, use 10B tokens from FALCON as per paper
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1%]")  # Small subset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # Causal LM
    return tokenized_dataset, data_collator

def continued_pretraining_loss(outputs, labels):
    """Standard cross-entropy loss for continued pretraining."""
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/marcos/BitNet/HY-MT1.5-1.8B"
    
    # Load student after Stage-1 (already with BitLinear and SubLN)
    student = init_bitnet_student(model_path, device)
    student.train()
    student.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset: small corpus as per Stage-2
    train_dataset, data_collator = prepare_corpus(tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    # Optimizer and scheduler (as per paper, standard settings)
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)
    
    num_epochs = 1  # In practice, train until convergence or 10B tokens
    step = 0
    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = student(**batch)
            loss = continued_pretraining_loss(outputs, batch["labels"])
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}")
            
            if step >= 1000:  # Early stop for demo
                break
    
    # Save checkpoint for Stage-3
    student.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    print("Stage-2 Continued Pre-Training complete. Checkpoint saved.")

if __name__ == "__main__":
    main()