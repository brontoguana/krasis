import sys
import os
import torch
import numpy as np

# Add python dir to path
sys.path.insert(0, os.path.join(os.getcwd(), "python"))

import krasis

def test_reduce_sum_bf16():
    print("Testing reduce_sum_bf16...")
    engine = krasis.KrasisEngine(parallel=True)
    
    # 2 GPUs, 1024 elements each
    num_elements = 1024
    
    # Create synthetic partial results
    # Use float32 for reference summation
    data1 = torch.randn(num_elements, dtype=torch.bfloat16)
    data2 = torch.randn(num_elements, dtype=torch.bfloat16)
    
    expected = (data1.float() + data2.float()).to(torch.bfloat16)
    
    # Output buffer
    output = torch.zeros(num_elements, dtype=torch.bfloat16)
    
    # Raw pointers
    input_ptrs = [data1.data_ptr(), data2.data_ptr()]
    output_ptr = output.data_ptr()
    
    engine.reduce_sum_bf16(input_ptrs, output_ptr, num_elements)
    
    # Check results
    diff = (output.float() - expected.float()).abs().max().item()
    print(f"  Max diff: {diff}")
    
    # BF16 precision is ~0.004 relative
    assert diff < 0.05, f"Reduce sum failed: max diff {diff}"
    print("PASS: reduce_sum_bf16")

def test_cpu_hub():
    print("\nTesting CPUHubManager...")
    from krasis.model import CPUHubManager
    
    engine = krasis.KrasisEngine(parallel=True)
    hidden_size = 4096
    num_gpus = 2
    hub = CPUHubManager(num_gpus, hidden_size, engine)
    
    M = 128
    # Synthetic GPU tensors
    t1 = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda:0")
    t2 = torch.randn(M, hidden_size, dtype=torch.bfloat16, device="cuda:1")
    
    expected = (t1.cpu().float() + t2.cpu().float()).to(torch.bfloat16)
    
    # 1. Gather
    hub.gather([t1, t2])
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    
    # 2. Reduce
    hub.reduce_sum(M)
    
    # 3. Broadcast back to target tensors
    out1 = torch.zeros_like(t1)
    out2 = torch.zeros_like(t2)
    hub.broadcast([out1, out2])
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    
    # Verify
    diff1 = (out1.cpu().float() - expected.float()).abs().max().item()
    diff2 = (out2.cpu().float() - expected.float()).abs().max().item()
    
    print(f"  Broadcast 1 max diff: {diff1}")
    print(f"  Broadcast 2 max diff: {diff2}")
    
    assert diff1 < 0.05
    assert diff2 < 0.05
    print("PASS: CPUHubManager")

if __name__ == "__main__":
    test_reduce_sum_bf16()
    if torch.cuda.device_count() >= 2:
        test_cpu_hub()
    else:
        print("\nSKIP: CPUHubManager test needs 2 GPUs")
