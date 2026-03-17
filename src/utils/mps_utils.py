"""MPS (Metal Performance Shaders) utility functions."""

import torch
import numpy as np

def get_mps_device():
    """Get MPS device if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_mps(tensor):
    """Move tensor to MPS if available."""
    if torch.backends.mps.is_available():
        return tensor.to("mps")
    return tensor

def mps_safe_tensor_operation(func):
    """Decorator to safely handle MPS tensor operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "MPS" in str(e):
                print(f"MPS error: {e}. Falling back to CPU.")
                # Move all tensors to CPU and retry
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.device.type == "mps":
                        new_args.append(arg.cpu())
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.device.type == "mps":
                        new_kwargs[key] = value.cpu()
                    else:
                        new_kwargs[key] = value
                
                return func(*new_args, **new_kwargs)
            raise
    return wrapper

def optimize_mps_memory():
    """Optimize MPS memory usage."""
    if torch.backends.mps.is_available():
        # Clear cache
        torch.mps.empty_cache()
        
        # Set memory fraction if needed
        # torch.mps.set_per_process_memory_fraction(0.8)
        
        print("MPS memory optimized")

def benchmark_mps():
    """Benchmark MPS vs CPU performance."""
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return
    
    # Create test tensors
    size = 1000
    cpu_tensor = torch.randn(size, size)
    mps_tensor = cpu_tensor.to("mps")
    
    # Benchmark matrix multiplication
    import time
    
    # CPU benchmark
    start = time.time()
    for _ in range(10):
        _ = torch.mm(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start
    
    # MPS benchmark
    start = time.time()
    for _ in range(10):
        _ = torch.mm(mps_tensor, mps_tensor)
    torch.mps.synchronize()  # Wait for MPS to finish
    mps_time = time.time() - start
    
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"MPS time: {mps_time:.4f}s")
    print(f"Speedup: {cpu_time/mps_time:.2f}x")