"""
training/eval_sanity.py - Sanity Check for BitNet Student Translation Performance
Loads student checkpoint, performs zero-shot translation, compares with teacher, and computes embedding cosine similarity.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from .init_student import init_bitnet_student

def load_models(model_path: str, student_checkpoint: str, device: str = "cuda"):
    """Load teacher (original) and student (1.58-bit) models."""
    # Teacher: original HY-MT1.5-1.8B
    teacher = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Student: from final checkpoint (after all stages)
    student = AutoModelForCausalLM.from_pretrained(student_checkpoint, torch_dtype=torch.float16).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return teacher, student, tokenizer

def generate_translation(model, tokenizer, sentence, device):
    """Generate translation using greedy search."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy search
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def get_embedding(model, tokenizer, text, device):
    """Get mean-pooled embedding of the text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Mean pool the last hidden state
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
    return embedding.squeeze()

def translate_and_compare(teacher, student, tokenizer, sentences, device):
    """Perform zero-shot translation and compare qualitatively and quantitatively."""
    similarities = []
    for sentence in sentences:
        # Generate translations
        teacher_translation = generate_translation(teacher, tokenizer, sentence, device)
        student_translation = generate_translation(student, tokenizer, sentence, device)
        
        # Compute embedding cosine similarity
        teacher_emb = get_embedding(teacher, tokenizer, teacher_translation, device)
        student_emb = get_embedding(student, tokenizer, student_translation, device)
        cos_sim = F.cosine_similarity(teacher_emb.unsqueeze(0), student_emb.unsqueeze(0)).item()
        similarities.append(cos_sim)
        
        # Print qualitative comparison
        print(f"Original: {sentence}")
        print(f"Teacher: {teacher_translation}")
        print(f"Student: {student_translation}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
        print("-" * 50)
    
    avg_sim = sum(similarities) / len(similarities)
    print(f"Average Cosine Similarity: {avg_sim:.4f}")
    return avg_sim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/marcos/BitNet/HY-MT1.5-1.8B"
    student_checkpoint = "/home/marcos/BitNet/student_final_checkpoints"
    
    teacher, student, tokenizer = load_models(model_path, student_checkpoint, device)
    
    sentences = [
        "The technology is advancing fast.",
        "I am learning to train models.",
        "This is a test sentence for translation.",
        "Machine learning is fascinating.",
        "How does the weather look today?"
    ]
    
    avg_sim = translate_and_compare(teacher, student, tokenizer, sentences, device)
    
    # Sanity check
    if avg_sim > 0.8:
        print("Sanity check passed: Student maintains semantic fidelity.")
    else:
        print("Sanity check failed: Further distillation needed.")

if __name__ == "__main__":
    main()