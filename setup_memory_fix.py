#!/usr/bin/env python3
"""
Script to set up memory management environment variables for better CUDA memory handling.
"""

import os
import sys

def setup_memory_environment():
    """Set up environment variables for better CUDA memory management."""
    
    # Set PyTorch CUDA memory allocation configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set memory fraction to prevent OOM
    os.environ['CUDA_MEMORY_FRACTION'] = '0.9'
    
    # Enable memory pool
    os.environ['PYTORCH_CUDA_MEMORY_POOL'] = 'True'
    
    print("✅ Memory management environment variables set:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_MEMORY_FRACTION: {os.environ.get('CUDA_MEMORY_FRACTION')}")
    print(f"  PYTORCH_CUDA_MEMORY_POOL: {os.environ.get('PYTORCH_CUDA_MEMORY_POOL')}")

if __name__ == "__main__":
    setup_memory_environment() 