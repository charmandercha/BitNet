#!/usr/bin/env python3
"""
run_bitdistill.py - Complete BitDistill Pipeline Runner
Executes the three-stage BitDistill process as per Microsoft paper.
"""

import torch
import argparse
import os
from training.init_student import init_bitnet_student
from training.distill import stage2_continue_pretrain, stage3_distillation
from training.eval_sanity import main as run_sanity_check

def main():
    parser = argparse.ArgumentParser(description="BitDistill Pipeline")
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "all"], default="all",
                       help="Which stage to run (1: SubLN, 2: Continue Pre-train, 3: Distillation)")
    parser.add_argument("--model_path", type=str, default="/home/marcos/BitNet/HY-MT1.5-1.8B",
                       help="Path to teacher model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation")
    
    args = parser.parse_args()
    
    if args.eval_only:
        print("Running evaluation only...")
        run_sanity_check()
        return
    
    print(f"BitDistill Pipeline on {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Stage: {args.stage}")
    print("-" * 50)
    
    if args.stage in ["1", "all"]:
        print("Stage-1: Model Refinement with SubLN")
        student = init_bitnet_student(args.model_path, args.device)
        print("Stage-1 complete: SubLN inserted and BitLinear layers created")
        print()
    
    if args.stage in ["2", "all"]:
        print("Stage-2: Continue Pre-training")
        stage2_continue_pretrain()
        print("Stage-2 complete: Model adapted to 1.58-bit representation")
        print()
    
    if args.stage in ["3", "all"]:
        print("Stage-3: Distillation-based Fine-tuning")
        stage3_distillation()
        print("Stage-3 complete: Final student model ready")
        print()
    
    print("BitDistill Pipeline Complete!")
    print("Final model saved at: /home/marcos/BitNet/student_final_checkpoints")
    
    # Run sanity check only if student_final_checkpoints exists
    import os
    if os.path.exists("/home/marcos/BitNet/student_final_checkpoints/pytorch_model.bin"):
        print("\nRunning sanity check...")
        run_sanity_check()
    else:
        print("\nSkipping sanity check - student_final_checkpoints not found.")
        print("Run 'python run_bitdistill.py --stage all' to create final checkpoints.")

if __name__ == "__main__":
    main()