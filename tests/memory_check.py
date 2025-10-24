"""
Memory leak detection utility for local debugging.
Only use for local testing, NOT for cloud training (performance impact).

This module provides:
- GPU memory tracking before/after operations
- Cumulative memory growth monitoring
- Memory leak warning threshold
"""

import torch
import gc
from typing import Optional, Callable, Dict, Tuple
import psutil
import os


class GPUMemoryTracker:
    """Track GPU memory usage across operations."""
    
    def __init__(self, device: torch.device = None, warn_threshold_mb: float = 100.0):
        """
        Args:
            device (torch.device): Target GPU device.
            warn_threshold_mb (float): Warn if memory growth exceeds this in MB.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.warn_threshold_mb = warn_threshold_mb
        self.baseline_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.iteration_memories = []
    
    def reset(self):
        """Reset tracking and clear cache."""
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.baseline_memory_mb = torch.cuda.memory_allocated() / 1e6
            self.peak_memory_mb = self.baseline_memory_mb
            self.iteration_memories = []
    
    def get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1e6
    
    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e6
    
    def record_iteration(self, iteration: int) -> Dict[str, float]:
        """Record memory after an iteration."""
        current_mb = self.get_memory_mb()
        peak_mb = self.get_peak_memory_mb()
        
        self.iteration_memories.append({
            'iteration': iteration,
            'current_mb': current_mb,
            'peak_mb': peak_mb,
            'growth_mb': current_mb - self.baseline_memory_mb,
        })
        
        return self.iteration_memories[-1]
    
    def check_leak(self, iteration: int, verbose: bool = True) -> bool:
        """
        Check if memory is leaking based on growth trend.
        
        Args:
            iteration (int): Current iteration number.
            verbose (bool): Print warnings.
        
        Returns:
            bool: True if leak detected (growth > threshold), False otherwise.
        """
        if len(self.iteration_memories) < 2:
            return False
        
        current_growth_mb = self.iteration_memories[-1]['growth_mb']
        is_leaking = current_growth_mb > self.warn_threshold_mb
        
        if is_leaking and verbose:
            print(f"⚠️  [Iteration {iteration}] Potential memory leak detected!")
            print(f"   Memory growth: {current_growth_mb:.2f} MB (threshold: {self.warn_threshold_mb:.2f} MB)")
            print(f"   Current usage: {self.iteration_memories[-1]['current_mb']:.2f} MB")
            print(f"   Peak usage: {self.iteration_memories[-1]['peak_mb']:.2f} MB")
        
        return is_leaking
    
    def print_summary(self):
        """Print memory tracking summary."""
        if not self.iteration_memories:
            print("No iteration data recorded.")
            return
        
        memories = self.iteration_memories
        growth_list = [m['growth_mb'] for m in memories]
        
        print("\n" + "="*70)
        print("GPU Memory Tracking Summary")
        print("="*70)
        print(f"Baseline memory: {self.baseline_memory_mb:.2f} MB")
        print(f"Peak memory: {max(m['peak_mb'] for m in memories):.2f} MB")
        print(f"Final memory: {memories[-1]['current_mb']:.2f} MB")
        print(f"Total growth: {growth_list[-1]:.2f} MB")
        print(f"Average growth/iteration: {sum(growth_list) / len(growth_list):.2f} MB")
        print(f"Max growth in single iteration: {max(growth_list):.2f} MB")
        
        # Check trend
        if len(growth_list) > 5:
            recent_growth = sum(growth_list[-5:]) / 5
            early_growth = sum(growth_list[:5]) / min(5, len(growth_list))
            print(f"Early avg growth/iter: {early_growth:.2f} MB")
            print(f"Recent avg growth/iter: {recent_growth:.2f} MB")
            
            if recent_growth > early_growth * 1.5:
                print("⚠️  WARNING: Memory growth is accelerating! Likely leak.")
            elif recent_growth < 10.0:
                print("✅ No obvious memory leak detected.")
        
        print("="*70 + "\n")


def track_memory(func: Callable) -> Callable:
    """
    Decorator to track memory usage of a function.
    
    Usage:
        @track_memory
        def my_test_func():
            ...
    """
    def wrapper(*args, **kwargs):
        tracker = GPUMemoryTracker()
        tracker.reset()
        
        print(f"\n[Memory Tracking] Running: {func.__name__}")
        print(f"Initial memory: {tracker.get_memory_mb():.2f} MB")
        
        result = func(*args, **kwargs)
        
        print(f"Final memory: {tracker.get_memory_mb():.2f} MB")
        print(f"Peak memory: {tracker.get_peak_memory_mb():.2f} MB")
        
        return result
    
    return wrapper


def batch_forward_memory_check(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 10,
    backward: bool = False,
) -> Tuple[Dict, bool]:
    """
    Run multiple forward passes and check for memory leaks.
    
    Args:
        model (torch.nn.Module): Model to test.
        input_tensor (torch.Tensor): Dummy input tensor.
        num_iterations (int): Number of iterations.
        backward (bool): Whether to include backward pass.
    
    Returns:
        Tuple[Dict, bool]: (memory_stats, has_leak)
    """
    device = next(model.parameters()).device
    tracker = GPUMemoryTracker(device=device, warn_threshold_mb=150.0)
    tracker.reset()
    
    print(f"\n[Memory Check] Running {num_iterations} iterations (backward={backward})")
    print(f"Input shape: {input_tensor.shape}, Device: {device}")
    
    has_leak = False
    
    with torch.no_grad() if not backward else torch.enable_grad():
        for i in range(num_iterations):
            # Forward pass
            output = model(input_tensor)
            
            if backward and output.requires_grad:
                loss = output.mean()
                loss.backward()
            
            # Record memory
            stats = tracker.record_iteration(i)
            leak_detected = tracker.check_leak(i, verbose=(i % 3 == 0))
            has_leak = has_leak or leak_detected
            
            # Clean up
            del output
            if backward:
                model.zero_grad()
            
            if i % 3 == 0:
                print(f"  Iter {i}: {stats['current_mb']:.2f} MB (growth: {stats['growth_mb']:.2f} MB)")
    
    tracker.print_summary()
    
    return {
        'baseline_mb': tracker.baseline_memory_mb,
        'peak_mb': tracker.peak_memory_mb,
        'final_mb': tracker.iteration_memories[-1]['current_mb'] if tracker.iteration_memories else 0,
        'total_growth_mb': tracker.iteration_memories[-1]['growth_mb'] if tracker.iteration_memories else 0,
    }, has_leak