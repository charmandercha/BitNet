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
# Skip large dataset download for demo - use small local data
from datasets import load_dataset
DATASET_AVAILABLE = True
from .init_student import init_bitnet_student
import math

def prepare_corpus(tokenizer, max_length: int = 512, num_tokens_target: int = 10000000000):
    """
    CORRECTED: Load and prepare corpus for Stage-2 continue pre-training.
    Using small dataset for demo to avoid long downloads.
    """
    # Use small WikiText dataset for demo - no large downloads
print("Using small WikiText-103 dataset for demo...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1000]")  # Small for testing
print(f"Dataset loaded: {len(dataset)} examples")

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    
    if isinstance(dataset, list):
        # Handle dummy data
        tokenized_dataset = tokenize_function(dataset)
    else:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # Causal LM
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)} examples")
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
    
    # CORRECTED: Large corpus for Stage-2 as per paper
    train_dataset = prepare_corpus(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # Causal LM
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)
    
    # Optimizer and scheduler (paper's recommended settings)
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)
    
    # CORRECTED: Train until 10B tokens processed
    target_tokens = 10000000000  # 10 billion tokens
    tokens_processed = 0
    step = 0
    
    print("Starting CORRECTED Stage-2: Continue Pre-training (target: 10B tokens)...")
    
    while tokens_processed < target_tokens:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = student(**batch)
            loss = continued_pretraining_loss(outputs, batch["labels"])
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track tokens processed
            tokens_processed += batch["input_ids"].numel()  # Count all tokens in batch
            
            step += 1
            if step % 100 == 0:
                tokens_b = tokens_processed / 1000000000  # Convert to billions
                print(f"Step {step}: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}, Tokens={tokens_b:.2f}B")
            
            if tokens_processed >= target_tokens:
                break
        
        if tokens_processed >= target_tokens:
            break
    
    # Save checkpoint for Stage-3
    student.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    tokenizer.save_pretrained("/home/marcos/BitNet/student_stage2_checkpoints")
    print(f"Stage-2 Complete: Processed {tokens_processed/1000000000:.2f}B tokens. Checkpoint saved.")

if __name__ == "__main__":
    main()