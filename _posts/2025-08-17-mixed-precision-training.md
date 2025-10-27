---
layout: distill
title: Mixed Precision Training in PyTorch
description: A comprehensive guide to mixed precision training in PyTorch covering torch.cuda.amp, gradient scaling, and performance optimization
date: 2025-08-17
categories: machine-learning
tags: [mixed-precision, pytorch, amp, training]
authors:
  - name: Nitin Singh
    url: https://nitinsinghgit.github.io
bibliography: 2025-08-17-mixed-precision-training.bib
---

# Mixed Precision Training in PyTorch

## Overview

Training large deep learning models requires massive computational resources. Mixed precision training combines 16-bit (FP16) and 32-bit (FP32) floating-point arithmetic to speed up training and reduce memory usage—without sacrificing accuracy. Modern GPUs (Volta, Turing, Ampere, Hopper) are optimized for FP16 operations, making mixed precision the default choice for production training.

The main benefits are:

1. **Memory Reduction:** FP16 uses half the memory of FP32, allowing larger batch sizes
2. **Faster Training:** Modern GPUs have specialized FP16 units
3. **Maintained Accuracy:** Critical operations remain in FP32 to preserve numerical stability
4. **Easy Implementation:** PyTorch's Automatic Mixed Precision (AMP) handles most complexity

## Why AMP (Automatic Mixed Precision) Matters

`torch.cuda.amp` is the *recommended* way to do mixed precision training in PyTorch. It automatically handles the complexity of mixed precision training by:

- Automatically casting operations to FP16 where beneficial, while keeping critical operations in FP32.
- Handling gradient scaling to prevent underflow in FP16 operations.
- Providing seamless integration with existing PyTorch training loops.

## Mixed Precision Training

### 1. Basic AMP Implementation

Simple to use and provides significant speedup. The key components are `autocast` for automatic casting and `GradScaler` for gradient scaling.

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Initialize model and optimizer
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).cuda()

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Initialize GradScaler for gradient scaling
scaler = GradScaler()

# Training loop with mixed precision
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Why Gradient Scaling?

FP16 has a narrower dynamic range than FP32, which can cause gradients to underflow. 
**Gradient scaling** multiplies the loss before backpropagation and rescales gradients afterward, preventing underflow. 
PyTorch's `GradScaler` automates this process.

### 1. Custom Autocast Context

```python
# Enable FP16 by default, but keep critical ops in FP32
with autocast():
    x = model_fp16(data)

with autocast(enabled=False):
    x = model_fp32(x)  # e.g., numerically sensitive operations
```

### 2. Model-Specific AMP

```python
class MixedPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_layer = nn.Linear(512, 256)
        self.fp32_layer = nn.Linear(256, 10)

    def forward(self, x):
        with autocast():
            x = torch.relu(self.fp16_layer(x))
        with autocast(enabled=False):
            x = self.fp32_layer(x)
        return x
```

## Memory and Performance Comparison

| Training Mode | Memory Usage | Training Speed | Accuracy |
|---------------|-------------|----------------|----------|
| FP32 Only | 100% (baseline) | 1× | Reference |
| Mixed Precision | ~50–60% | 1.5–2× | Same |
| FP16 Only | ~50% | 2–3× | Often worse |

## Best Practices

Follow these guidelines to ensure successful mixed precision training:

- **Always use GradScaler:** Handles loss scaling automatically
- **Use gradient clipping:** Prevents exploding gradients
- **Profile speed & memory:** Not all models benefit equally
- **AMP works with DDP & FSDP:** Fully compatible with distributed training

## Common Issues and Solutions

### 1. Loss Explosion

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

### 2. Model Checkpointing

```python
# Save
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
}, "checkpoint.pth")

# Load
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scaler.load_state_dict(checkpoint['scaler'])
```

## Integration with Distributed Training

Mixed precision training delivers **1.5–3× faster training** and **~50% memory savings** with
minimal code changes. For large-scale vision and NLP models, it's often the difference between
fitting on a single GPU and needing model parallelism.

If you're not already using `torch.cuda.amp`, you're leaving performance on the table.
Try enabling it in your next training run and benchmark the speedup.
