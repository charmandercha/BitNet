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

def prepare_corpus(tokenizer, max_length: int = 512, num_tokens_target: int = 10000000000):
    """
    CORRECTED: Load and prepare FALCON corpus as per BitDistill paper requirements.
    Target: 10B tokens for proper Stage-2 continue pre-training.
    """
    try:
        # PRIMARY: Use FALCON-refinedweb dataset as per paper
        print("Loading FALCON-refinedweb dataset...")
        dataset = load_dataset("tiiuae/falcon-refinedweb", split="train")
        print(f"FALCON dataset loaded: {len(dataset)} examples")
        
        # Calculate samples needed for ~10B tokens (assuming avg 200 tokens per example)
        target_samples = num_tokens_target // 200
        if len(dataset) > target_samples:
            dataset = dataset.select(range(target_samples))
        
    except Exception as e:
        print(f"FALCON dataset not available: {e}")
        print("Falling back to WikiText-103 (Note: Performance will be suboptimal)")
        
        # FALLBACK: Use larger WikiText subset for demo
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:5%]")  # Larger fallback
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
    
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
    train_dataset, data_collator = prepare_corpus(tokenizer)
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