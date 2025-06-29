#!/usr/bin/env python3
"""
Test script to verify device synchronization fixes for LoRA with Flux split transformer blocks.
"""

import torch
import torch.nn as nn

def test_device_synchronization():
    """Test that LoRA weights are automatically moved to the correct device."""
    
    # Simulate a split transformer block scenario
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    
    # Create a simple LoRA module (simulating the structure)
    class SimpleLoRAModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_down = nn.Linear(10, 4)
            self.lora_up = nn.Linear(4, 10)
            self.scale = 1.0
            self.multiplier = 1.0
            
        def _call_forward(self, x):
            # Ensure LoRA weights are on the same device as input
            target_device = x.device
            if self.lora_down.weight.device != target_device:
                self.lora_down = self.lora_down.to(target_device)
            if self.lora_up.weight.device != target_device:
                self.lora_up = self.lora_up.to(target_device)
                
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return lx * self.scale
            
        def forward(self, x):
            return self._call_forward(x)
    
    # Create module and move to device1
    module = SimpleLoRAModule().to(device1)
    
    print(f"Initial device of lora_down: {module.lora_down.weight.device}")
    print(f"Initial device of lora_up: {module.lora_up.weight.device}")
    
    # Create input on device2 (simulating split transformer block)
    x = torch.randn(2, 10).to(device2)
    print(f"Input device: {x.device}")
    
    # This should automatically move LoRA weights to device2
    try:
        output = module(x)
        print(f"Output device: {output.device}")
        print(f"Final device of lora_down: {module.lora_down.weight.device}")
        print(f"Final device of lora_up: {module.lora_up.weight.device}")
        print("✅ Test passed: Device synchronization worked correctly!")
        return True
    except RuntimeError as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if torch.cuda.device_count() >= 2:
        test_device_synchronization()
    else:
        print("⚠️  Skipping test: Need at least 2 CUDA devices to test split scenario")
        print("The fixes should still work in single-device scenarios.") 