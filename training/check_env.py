#!/usr/bin/env python3
"""
check_env.py - Environment check script for BitDistill implementation
Verifies PyTorch GPU support (ROCm), and key libraries for model distillation.
"""

import sys
import subprocess

def check_library(lib_name, import_name=None):
    """Check if a library is installed and can be imported."""
    if import_name is None:
        import_name = lib_name
    try:
        __import__(import_name)
        print(f"✓ {lib_name} is installed")
        return True
    except ImportError:
        print(f"✗ {lib_name} is not installed")
        return False

def check_pytorch_gpu():
    """Check PyTorch version and GPU support."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✓ CUDA is available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            print("✓ ROCm is available")
            print(f"  HIP version: {torch.version.hip}")
        else:
            print("✗ No GPU support detected (neither CUDA nor ROCm)")
            return False
        return True
    except ImportError:
        print("✗ PyTorch is not installed")
        return False

def check_transformers():
    """Check transformers library."""
    return check_library("transformers")

def check_accelerate():
    """Check accelerate library."""
    return check_library("accelerate")

def check_triton():
    """Check Triton compatibility."""
    try:
        import triton
        print("✓ Triton is installed")
        try:
            # Check if compatible with GPU
            import torch
            if torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip):
                print("✓ Triton compatible with GPU")
            else:
                print("⚠ Triton installed but no GPU detected")
        except Exception as e:
            print(f"⚠ Triton check failed: {e}")
        return True
    except ImportError:
        print("✗ Triton is not installed")
        return False

def main():
    print("=== BitDistill Environment Check ===\n")
    
    all_good = True
    
    print("1. Checking PyTorch and GPU support:")
    all_good &= check_pytorch_gpu()
    print()
    
    print("2. Checking key libraries:")
    all_good &= check_transformers()
    all_good &= check_accelerate()
    all_good &= check_triton()
    print()
    
    if all_good:
        print("🎉 Environment looks good for BitDistill implementation!")
        print("Ready to proceed with model distillation.")
    else:
        print("❌ Some dependencies are missing. Please install them:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6  # For ROCm")
        print("  pip install transformers accelerate")
        sys.exit(1)

if __name__ == "__main__":
    main()