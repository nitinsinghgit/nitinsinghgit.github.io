---
layout: distill
title: Distributed Training in PyTorch
description: A comprehensive guide to distributed training in PyTorch covering DistributedDataParallel (DDP), DataParallel, and model parallel approaches
date: 2025-08-16
categories: machine-learning
tags: [distributed-training, pytorch, ddp, fsdp]
authors:
  - name: Nitin Singh
    url: https://nitinsinghgit.github.io
bibliography: 2025-08-16-distributed-training-pytorch.bib
---

# Distributed Training in PyTorch

## Overview

Scaling deep learning training requires distributing work across multiple GPUs and often across multiple nodes. For small models, a single GPU is sufficient. But for large-scale models with billions of parameters or training datasets with billions of samples, distributed training is essential. PyTorch provides several strategies to achieve this.

The main approaches are:

1. **Data Parallel Training** (replicate model, split data)
2. **Model Parallel Training** (split model across GPUs)

## Why DDP (DistributedDataParallel) Matters

`torch.nn.parallel.DistributedDataParallel` (DDP) is the *recommended* way to do data-parallel training in PyTorch. It outperforms `DataParallel` because:

- Each process runs its own copy of the model on a dedicated GPU, avoiding Python GIL bottlenecks.
- Communication (gradient synchronization) is implemented in C++ with NCCL, enabling overlap between computation and communication.
- It scales across multiple machines seamlessly.

## Data Parallel Training

### 1. DataParallel (Single Machine, Multiple GPUs)

Easy to use but slower. Avoid for large models or multi-node training.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to('cuda')
```

### 2. DistributedDataParallel (DDP)

The preferred way to scale training. Each process gets one GPU, data is split using `DistributedSampler`, and gradients are synchronized via NCCL backend.

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    torch.cuda.set_device(rank)
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling each epoch
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    cleanup()

# Run with:
# torchrun --nproc_per_node=4 train.py
```

### Common Pitfalls in DDP

- **Not using DistributedSampler**: Without it, each process sees the entire dataset → duplicated work.
- **Forgetting sampler.set_epoch(epoch)**: Leads to identical shuffling each epoch.
- **find_unused_parameters=True**: Needed if some branches of the model are not always used.
- **CUDA_VISIBLE_DEVICES**: Restrict which GPUs are visible before launching with `torchrun`.

## Model Parallel Training

When a model doesn't fit on one GPU, we shard it. Techniques include:

- **Pipeline Parallelism**: Split layers into stages.
- **FSDP**: Shard parameters and optimizer states.
- **Tensor Parallelism**: Split individual layers across GPUs.

## Fully Sharded Data Parallel (FSDP)

FSDP is PyTorch's most powerful distributed training strategy for large models.
It shards model parameters, gradients, and optimizer states across GPUs, reducing
memory usage and enabling training of models that don't fit on a single GPU.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        return self.net(x)

def train(rank, world_size, epochs=5):
    setup(rank, world_size)

    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = LargeModel().to(rank)

    # Wrap with FSDP (no mixed precision)
    model = FSDP(model,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    cleanup()

# Run with:
# torchrun --nproc_per_node=4 fsdp_train.py
```

### Key Notes on FSDP

- **ShardingStrategy.FULL_SHARD**: shards parameters, gradients, and optimizer states across GPUs (memory efficient).
- **MixedPrecision**: lowers memory and speeds up training by using FP16 for parameters and gradients.
- **Gradient checkpointing**: can be combined with FSDP for even larger models.
- **Launch with torchrun**: one process per GPU is still required.

## Best Practices

- **Prefer DDP over DataParallel**: DDP avoids GIL overhead and scales across machines.
- **Match GPUs to processes**: 1 process per GPU → better performance.
- **Overlap communication with compute**: DDP does this automatically, but ensure large enough batch sizes.
- **Use gradient accumulation**: For very large effective batch sizes.
- **Enable mixed precision**: `torch.cuda.amp` for faster training and lower memory.
- **Profile with Nsight / PyTorch profiler**: Detect communication bottlenecks.

## Performance Comparison

| Method | Memory Usage | Communication Overhead | Scalability | Use Case |
|--------|-------------|----------------------|-------------|----------|
| DataParallel | High (replicated) | Low | Single machine only | Small models, prototyping |
| DistributedDataParallel (DDP) | High (replicated) | Medium, overlaps with compute | Multi-machine, scalable | Most common choice |
| FSDP | Low (sharded) | Medium–High | Multi-machine | Large models, memory efficient |
| Pipeline Parallel | Low (split layers) | High | Multi-machine | Very large models (GPT, LLaMA) |
