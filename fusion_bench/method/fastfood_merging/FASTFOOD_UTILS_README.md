# Fastfood Utilities Module

## Overview

The `fastfood_utils.py` module provides a comprehensive set of utility functions for Fastfood/SRHT-based model merging. This module centralizes common operations used across different fastfood merging algorithms, promoting code reuse and maintainability.

## Module Structure

The module is organized into several functional categories:

### 1. Fastfood / SRHT Core Operations

#### `next_pow2(n: int) -> int`
Compute the next power of 2 greater than or equal to n.

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import next_pow2

result = next_pow2(100)  # Returns 128
result = next_pow2(64)   # Returns 64
```

#### `fwht_inplace_ortho(x: Tensor) -> Tensor`
In-place orthonormal Fast Walsh-Hadamard Transform (FWHT) along the last dimension. The transform is scaled by 1/sqrt(n) to maintain orthonormality.

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import fwht_inplace_ortho

x = torch.randn(10, 128)
x_transformed = fwht_inplace_ortho(x)  # Applies FWHT in-place
```

#### `seed_from_string(s: str) -> int`
Generate a deterministic seed from a string using MD5 hash. Useful for ensuring reproducibility across runs.

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import seed_from_string

seed = seed_from_string("layer.0.attention")  # Deterministic seed for this layer
```

#### `create_fastfood_ops(global_dim, proj_dim, *, seed_key, device, use_G)`
Build Fastfood/SRHT projection operators for dimension reduction.

**Parameters:**
- `global_dim`: Original dimension D
- `proj_dim`: Target projection dimension m
- `seed_key`: String used to seed random generation (for reproducibility)
- `device`: Torch device for computation
- `use_G`: Whether to use Gaussian scaling (G matrix)

**Returns:**
- Tuple of `(fwd, lift)` functions for projection and lifting

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import create_fastfood_ops
import torch

fwd, lift = create_fastfood_ops(
    global_dim=1000,
    proj_dim=100,
    seed_key="layer_1",
    device=torch.device("cuda"),
    use_G=True
)

# Project high-dimensional vector to low dimension
x = torch.randn(1000, device="cuda")
x_proj = fwd(x)  # Shape: (100,)

# Lift back to original dimension
x_lifted = lift(x_proj)  # Shape: (1000,)
```

### 2. TIES Merging Functions

TIES (Task-wise Interference Elimination by Sign) merging resolves parameter conflicts by electing dominant signs and performing disjoint aggregation.

#### `resolve_zero_signs(sign_to_mult: Tensor, method: str = "majority") -> Tensor`
Resolve zero signs in a tensor by majority or minority rule.

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import resolve_zero_signs

signs = torch.tensor([-1, 0, 1, -1, 0])
resolved = resolve_zero_signs(signs, method="majority")  # Zeros become -1 (majority)
```

#### `resolve_sign(v: Tensor) -> Tensor`
Resolve the sign of a tensor by majority rule across tasks.

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import resolve_sign

# Task vectors: [K=3 tasks, 5 parameters]
task_vectors = torch.tensor([
    [1.0, -2.0, 3.0, -1.0, 0.0],
    [2.0, -1.0, -1.0, -2.0, 1.0],
    [1.5, -3.0, 0.0, -1.5, 0.5]
])

elected_signs = resolve_sign(task_vectors)  # Shape: (5,)
```

#### `ties_disjoint_merge(v: Tensor, merge_func: str, sign_to_mult: Tensor) -> Tensor`
Perform TIES disjoint merging using a specified merge function. Only parameters that agree with the elected sign are merged.

**Parameters:**
- `v`: Input tensor [K, ...] where K is number of tasks
- `merge_func`: "mean", "sum", or "max"
- `sign_to_mult`: Tensor with elected signs for merging

#### `ties_merge_subspace(U: Tensor, ties_merge_func: str = "sum", weights: List[float] | None = None) -> Tensor`
Complete TIES merging pipeline in subspace (no trimming, only elect + disjoint merge).

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import ties_merge_subspace

# Task vectors in subspace: [K=3 tasks, 100 dimensions]
U = torch.randn(3, 100)

merged = ties_merge_subspace(
    U,
    ties_merge_func="mean",
    weights=[1.0, 1.5, 0.5]  # Task importance weights
)
```

### 3. EMA (Exponential Moving Average) Merging Functions

EMA merging processes tasks sequentially with adaptive weighting based on alignment and scale.

#### `ema_adaptive_beta(z_acc, z_new, gamma=1.2, w_c=0.6, w_s=0.4, eps=1e-8) -> float`
Compute adaptive β_t for EMA based on alignment and scale between accumulator and new task.

**Parameters:**
- `z_acc`: Current accumulator vector
- `z_new`: New task vector to incorporate
- `gamma`: Sigmoid scaling factor (default 1.2)
- `w_c`: Weight for cosine alignment term (default 0.6)
- `w_s`: Weight for scale ratio term (default 0.4)
- `eps`: Small constant for numerical stability

**Returns:**
- β_t ∈ [0,1]: mixing coefficient for EMA update

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import ema_adaptive_beta

z_acc = torch.randn(100)
z_new = torch.randn(100)

beta = ema_adaptive_beta(z_acc, z_new, gamma=1.5, w_c=0.7, w_s=0.3)
# Use beta for EMA update: z_acc = beta * z_acc + (1-beta) * z_new
```

#### `ema_merge_subspace(U, task_order="given", ema_gamma=1.2, ema_w_c=0.6, ema_w_s=0.4, weights=None, custom_order=None) -> Tensor`
Complete EMA merging pipeline in subspace with adaptive β_t.

**Parameters:**
- `U`: [K, ..., M] stacked task vectors in subspace
- `task_order`: "given", "random", "cosine_similarity", or "custom"
- `ema_gamma`: Sigmoid scaling factor for β computation
- `ema_w_c`: Weight for cosine alignment term
- `ema_w_s`: Weight for scale ratio term
- `weights`: Task importance weights
- `custom_order`: List of task indices when task_order="custom"

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import ema_merge_subspace

# Task vectors in subspace: [K=5 tasks, 100 dimensions]
U = torch.randn(5, 100)

# Merge with cosine similarity ordering
merged = ema_merge_subspace(
    U,
    task_order="cosine_similarity",
    ema_gamma=1.2,
    weights=[1.0, 1.0, 1.5, 0.8, 1.2]
)

# Merge with custom order
merged_custom = ema_merge_subspace(
    U,
    task_order="custom",
    custom_order=[2, 0, 4, 1, 3]  # Process tasks in this order
)
```

### 4. Zero-Aware Aggregation Functions

#### `zero_aware_aggregate(U, merge_func, weights=None, **kwargs) -> Tensor`
Unified zero-aware aggregation function supporting multiple merge strategies.

**Supported merge functions:**
- `'sum'`: Elementwise sum (weights optional)
- `'mean'`: Elementwise mean over *nonzero* contributors only
- `'max'`: Elementwise argmax by |u_k|; zeros preserved
- `'signmax'`: Per position, pick dominant sign, then largest |u_k| with that sign
- `'ema'`: Exponential Moving Average with adaptive β_t
- `'ties_sum'`: TIES merging with sum aggregation
- `'ties_mean'`: TIES merging with mean aggregation
- `'ties_max'`: TIES merging with max aggregation

**Example:**
```python
import torch
from fusion_bench.method.fastfood_merging.fastfood_utils import zero_aware_aggregate

# Task vectors: [K=3 tasks, 10 parameters]
U = torch.randn(3, 10)

# Different merge strategies
merged_sum = zero_aware_aggregate(U, "sum", weights=[1.0, 1.5, 0.5])
merged_mean = zero_aware_aggregate(U, "mean")
merged_max = zero_aware_aggregate(U, "max")
merged_signmax = zero_aware_aggregate(U, "signmax")
merged_ema = zero_aware_aggregate(
    U, "ema",
    ema_task_order="cosine_similarity",
    ema_gamma=1.2
)
merged_ties = zero_aware_aggregate(U, "ties_mean", weights=[1.0, 1.0, 1.5])
```

### 5. Helper Utilities

#### `layer_key(name: str) -> str`
Heuristic layer-grouping key for parameter names. Works for most Hugging Face transformer models by extracting the first 2-3 components.

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import layer_key

key = layer_key("model.layer.0.attention.query.weight")
# Returns: "model.layer.0"

key = layer_key("encoder.layer.5.output.dense.bias")
# Returns: "encoder.layer.5"
```

#### `compute_global_dim(state_dict: dict, keys: List[str]) -> int`
Compute the total number of parameters across specified keys.

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import compute_global_dim

state_dict = model.state_dict()
eligible_keys = [k for k in state_dict.keys() if 'weight' in k]
total_params = compute_global_dim(state_dict, eligible_keys)
print(f"Total parameters: {total_params:,}")
```

#### `normalize_weights(weights: List[float] | None, num_tasks: int) -> List[float]`
Normalize task weights to sum to 1. Returns uniform weights if None provided.

**Example:**
```python
from fusion_bench.method.fastfood_merging.fastfood_utils import normalize_weights

# Normalize custom weights
weights = normalize_weights([2.0, 3.0, 5.0], num_tasks=3)
# Returns: [0.2, 0.3, 0.5]

# Generate uniform weights
weights = normalize_weights(None, num_tasks=5)
# Returns: [0.2, 0.2, 0.2, 0.2, 0.2]
```

## Integration with Existing Code

The utilities module maintains backward compatibility with existing code through re-exports in the main merging modules:

```python
# In fastfood_merging.py and multi_scale_fastfood_merging.py
from .fastfood_utils import (
    EPS,
    create_fastfood_ops,
    zero_aware_aggregate,
    layer_key,
)

# Backward compatibility aliases
_fastfood_ops = create_fastfood_ops
_zero_aware_aggregate = zero_aware_aggregate
_layer_key = layer_key
```

This means existing code using `_fastfood_ops`, `_zero_aware_aggregate`, etc., will continue to work without modification.

## Best Practices

1. **Use descriptive seed_keys**: When creating Fastfood operators, use descriptive seed keys (e.g., layer names) for reproducibility.

2. **Choose appropriate merge functions**: 
   - Use `'sum'` or `'mean'` for simple averaging
   - Use `'signmax'` when you want to preserve task-specific directions
   - Use `'ema'` for sequential processing with adaptive weighting
   - Use `'ties_*'` methods to handle parameter conflicts

3. **Task ordering in EMA**: The order of tasks can significantly affect results:
   - `'given'`: Use modelpool order (default)
   - `'random'`: Reduce order bias
   - `'cosine_similarity'`: Process similar tasks first
   - `'custom'`: Full control over processing order

4. **Weight normalization**: Always normalize task weights to ensure stable merging behavior.

## Performance Considerations

- **FWHT is in-place**: The `fwht_inplace_ortho` function modifies tensors in-place for memory efficiency
- **Use appropriate block_rows**: When processing large tensors, use blocking to manage memory
- **Device placement**: Ensure all tensors are on the same device before merging

## Constants

- `EPS = 1e-12`: Small constant for numerical stability in divisions and comparisons

## See Also

- `FastfoodSubspaceMergeAlgorithm`: Main single-scale merging algorithm
- `MultiScaleFastfoodMergeAlgorithm`: Multi-scale merging algorithm
- Main documentation: `FASTFOOD_WRAPPER_README.md`
