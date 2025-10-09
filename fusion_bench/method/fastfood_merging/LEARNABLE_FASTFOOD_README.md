# Learnable Layer-wise Fastfood Merging

## Overview

This module implements a learnable variant of Fastfood/SRHT-based model merging where the projection ratio (subspace dimension) for each layer is **learned during test-time adaptation**. Unlike standard Fastfood merging which uses fixed projection ratios, this method optimizes the projection ratios to find the optimal subspace size for each layer.

## Key Innovation

The core insight is that **different layers prefer different subspace sizes for merging**. By making the projection ratios learnable parameters, we can automatically discover the optimal compression level for each layer.

### What Gets Learned

- **Projection Ratios**: A continuous value between 0 and 1 for each layer, determining the target dimension for the Fastfood projection
- **Only the projection ratios are trainable** - the merging process itself remains unchanged from the base Fastfood algorithm

### What Stays Fixed

- The merging function (sum, mean, TIES, etc.)
- The Fastfood/SRHT transformation operators
- The task weights
- All other hyperparameters

## Architecture

The implementation follows a clean, modular structure inspired by AdaMerging:

```
learnable_fastfood_merging.py         # Main algorithm (abstract base)
├── LearnableFastfoodMergingAlgorithm # Core algorithm with TTA
│   ├── construct_learnable_merged_model()
│   ├── test_time_adaptation()
│   └── Abstract methods (to be implemented by subclasses):
│       ├── on_test_time_adaptation_start()
│       ├── get_shuffled_test_loader_iter()
│       └── compute_logits()

clip_learnable_fastfood_merging.py    # CLIP-specific implementation
└── CLIPLearnableFastfoodMergingAlgorithm
    └── Implements CLIP-specific data loading and logits computation

learnable_fastfood_fusion.py          # Wrapper model
└── LearnableFastfoodMergedModel
    ├── Wraps the base model
    ├── Stores learnable projection_ratios parameter
    └── Dynamically merges weights in forward pass
```

## How It Works

### 1. Initialization

```python
# Initialize learnable projection ratios for each layer
projection_ratios = torch.full((num_layers,), init_proj_ratio)  # e.g., [0.1, 0.1, ..., 0.1]
projection_ratios = nn.Parameter(projection_ratios, requires_grad=True)
```

### 2. Forward Pass (Dynamic Merging)

For each forward pass:
1. Get current projection ratios (clamped to valid range)
2. For each layer:
   - Compute projection dimension: `proj_dim = int(layer_dim * proj_ratio)`
   - Create Fastfood operators with this dimension
   - Project task vectors to subspace
   - Merge in subspace (using fixed merge function)
   - Lift back to full dimension
3. Perform forward pass with merged weights

### 3. Test-Time Adaptation

```python
for step in range(max_steps):
    for task in tasks:
        # Get test batch (unlabeled data)
        batch = next(test_loader_iter(task))
        
        # Forward pass with current projection ratios
        logits = model(batch)
        
        # Minimize entropy (encourages confident predictions)
        loss = entropy_loss(logits)
        
        # Update projection ratios
        loss.backward()
        optimizer.step()
        
        # Re-merge with new ratios
        model.merge_weights()
```

### 4. Final Merging

After optimization, the learned projection ratios are used for one final merge, and the wrapper is unloaded to return a standard merged model.

## Usage

### CLIP Example

```bash
fusion_bench \
    method=learnable_fastfood/clip \
        method.init_proj_ratio=0.1 \
        method.lr=0.01 \
        method.max_steps=500 \
        method.save_projection_ratios=learned_ratios.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8
```

### Python API

```python
from fusion_bench.method.fastfood_merging import CLIPLearnableFastfoodMergingAlgorithm

algorithm = CLIPLearnableFastfoodMergingAlgorithm(
    init_proj_ratio=0.1,    # Start all layers at 10% projection
    lr=0.01,                 # Learning rate
    max_steps=500,           # Optimization steps
    clamp_ratios=True,       # Clamp to [min_proj_ratio, max_proj_ratio]
    min_proj_ratio=0.01,     # Minimum 1% projection
    max_proj_ratio=1.0,      # Maximum 100% projection
    merge_func="sum",        # Fixed merge function
    save_projection_ratios="learned_ratios.pt",
)

merged_model = algorithm.run(modelpool)
```

### Loading Pre-Learned Ratios

If you have pre-learned projection ratios, you can skip the test-time adaptation:

```python
# TODO: Add support for loading pre-learned ratios in future version
# Currently always performs test-time adaptation
```

## Configuration Options

### Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_proj_ratio` | float | 0.1 | Initial projection ratio for all layers |
| `lr` | float | 0.01 | Learning rate for optimizer |
| `max_steps` | int | 500 | Number of optimization steps |
| `optimizer` | str | "adam" | Optimizer type (currently only "adam") |
| `batch_size` | int | 16 | Batch size for test data |
| `num_workers` | int | 8 | Number of data loading workers |

### Projection Ratio Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clamp_ratios` | bool | True | Whether to clamp ratios to valid range |
| `min_proj_ratio` | float | 0.01 | Minimum allowed projection ratio |
| `max_proj_ratio` | float | 1.0 | Maximum allowed projection ratio |

### Fastfood Parameters (Fixed During Learning)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_func` | str | "sum" | Aggregation function (sum, mean, ties_sum, etc.) |
| `use_G` | bool | False | Use Gaussian scaling in Fastfood |
| `block_rows` | int | 8192 | Block size for processing |
| `weights` | List[float] | None | Task weights (None = uniform) |
| `scale` | float | 1.0 | Post-merge scaling factor |

## Implementation Details

### Wrapper Model

The `LearnableFastfoodMergedModel` class wraps the base model and handles:
- Storing learnable `projection_ratios` parameter
- Lazy merging (only recomputes when ratios change)
- Clamping ratios to valid range
- Delegating actual merging to a provided `merge_function`

### Merge Function

The merge function is a closure that captures the model state and performs merging:

```python
def merge_function(proj_ratios: Tensor) -> StateDictType:
    merged_sd = {}
    for layer_idx, layer_key in enumerate(layer_keys):
        proj_ratio = proj_ratios[layer_idx].item()
        # Perform Fastfood merging for this layer with proj_ratio
        # ...
    return merged_sd
```

This design keeps the merging logic separate from the wrapper, maintaining clean separation of concerns.

### Memory Efficiency

- Uses lazy merging (only recomputes when ratios change significantly)
- Processes layers in blocks to control memory usage
- Supports gradient accumulation across tasks

## Comparison with AdaMerging

| Aspect | AdaMerging | Learnable Fastfood |
|--------|------------|-------------------|
| **What's Learned** | Task weights per layer | Projection ratios per layer |
| **Merging Space** | Full parameter space | Compressed subspace |
| **Memory Usage** | High (stores all task vectors) | Lower (compression) |
| **Flexibility** | Learns contribution of each task | Learns optimal compression per layer |
| **Use Case** | When you want to adjust task importance | When you want optimal subspace size |

## Extending to Other Modalities

To implement for other model types (e.g., language models), create a subclass similar to `CLIPLearnableFastfoodMergingAlgorithm`:

```python
class MyModelLearnableFastfoodMergingAlgorithm(LearnableFastfoodMergingAlgorithm):
    def on_test_time_adaptation_start(self):
        # Setup task-specific components
        pass
    
    def get_shuffled_test_loader_iter(self, task: str):
        # Return data loader iterator
        pass
    
    def compute_logits(self, module, batch, task: str) -> Tensor:
        # Compute logits for your task
        pass
```

## Expected Behavior

### During Optimization

- Early layers typically converge to **lower projection ratios** (more compression)
- Later layers typically converge to **higher projection ratios** (less compression)
- The entropy loss should decrease, indicating more confident predictions
- Projection ratios should stabilize after sufficient optimization steps

### Final Results

- Learned ratios reveal which layers benefit from larger subspaces
- Can be analyzed to understand layer-wise merging dynamics
- Can potentially be transferred to similar model architectures

## Troubleshooting

### Issue: Ratios all converge to min/max bounds

**Solution**: Adjust `min_proj_ratio` and `max_proj_ratio` to give more room for optimization

### Issue: Loss doesn't decrease

**Solutions**:
- Increase learning rate (`lr`)
- Increase number of steps (`max_steps`)
- Check that test data is being loaded correctly
- Verify that zero-shot heads are set up properly (for CLIP)

### Issue: Out of memory

**Solutions**:
- Decrease `batch_size`
- Decrease `block_rows`
- Use gradient accumulation (future feature)

## Future Enhancements

Potential improvements:
- [ ] Support for loading pre-learned projection ratios
- [ ] Per-parameter projection ratios (instead of per-layer)
- [ ] Adaptive learning rate scheduling
- [ ] Support for other optimizers (SGD, AdamW, etc.)
- [ ] Gradient accumulation for larger effective batch sizes
- [ ] Regularization on projection ratios (e.g., L1 for sparsity)
- [ ] Multi-stage optimization (coarse-to-fine)

## References

This work builds upon:
- **Fastfood/SRHT**: Le et al., "Fastfood - Approximating Kernel Expansions in Loglinear Time"
- **AdaMerging**: Yang et al., "AdaMerging: Adaptive Model Merging for Multi-Task Learning" (ICLR 2024)
- **TIES**: Yadav et al., "TIES-Merging: Resolving Interference When Merging Models"

## Citation

If you use this method, please cite:

```bibtex
# TODO: Add citation once published
```
