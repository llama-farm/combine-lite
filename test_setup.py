#!/usr/bin/env python3
"""Test script to verify the AI Model Studio setup in both DEV and PROD modes."""

import torch
import sys
import argparse

def test_setup(dev_mode=False):
    print("=" * 50)
    print("AI Model Studio Setup Test")
    print("=" * 50)
    
    # Check mode
    mode = "DEV (ARM64/CPU)" if dev_mode else "PROD (CUDA GPU)"
    print(f"Mode: {mode}")
    
    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not dev_mode and torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    
    # Test tensor creation
    device = "cpu" if dev_mode else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    try:
        x = torch.randn(3, 3).to(device)
        print("✓ Successfully created tensor on device")
    except Exception as e:
        print(f"✗ Failed to create tensor: {e}")
        return False
    
    # Check imports
    print("\nChecking imports...")
    try:
        import gradio
        print("✓ Gradio imported successfully")
        
        import transformers
        print("✓ Transformers imported successfully")
        
        import peft
        print("✓ PEFT imported successfully")
        
        import datasets
        print("✓ Datasets imported successfully")
        
        import openai
        print("✓ OpenAI imported successfully")
        
        import boto3
        print("✓ Boto3 imported successfully")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test AI Model Studio setup')
    parser.add_argument('--dev', action='store_true', help='Test in development mode')
    args = parser.parse_args()
    
    success = test_setup(args.dev)
    sys.exit(0 if success else 1) 