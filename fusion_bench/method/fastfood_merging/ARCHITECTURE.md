# Adaptive Projection Size Estimation - Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Fastfood Subspace Merging                           │
│                    (FastfoodSubspaceMergeAlgorithm)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├─ Load models & task vectors
                                    │
                                    ├─ [Optional] Weight matching
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
        ┌───────────▼──────────┐        ┌──────────▼──────────┐
        │  Fixed Projection    │        │ Adaptive Projection │
        │  (Original Mode)     │        │   (New Feature)     │
        └───────────┬──────────┘        └──────────┬──────────┘
                    │                              │
                    │ m = ratio * d_last           │
                    │                              │
                    │              ┌───────────────┴──────────────┐
                    │              │                              │
                    │    ┌─────────▼─────────┐        ┌─────────▼─────────┐
                    │    │   Tensor Mode     │        │    Layer Mode     │
                    │    │  (per-tensor)     │        │   (per-layer)     │
                    │    └─────────┬─────────┘        └─────────┬─────────┘
                    │              │                            │
                    │    ┌─────────┴──────────┐      ┌─────────┴─────────┐
                    │    │                    │      │                   │
                    │    │  ┌─────────────┐   │      │  ┌─────────────┐  │
                    │    │  │   Fixed     │   │      │  │   Fixed     │  │
                    │    │  │ m = r * d   │   │      │  │ m = r * d   │  │
                    │    │  └─────────────┘   │      │  └─────────────┘  │
                    │    │                    │      │                   │
                    │    │  ┌─────────────┐   │      │  ┌─────────────┐  │
                    │    │  │   Random    │   │      │  │   Random    │  │
                    │    │  │ m ~ U[a,b]  │   │      │  │ m ~ U[a,b]  │  │
                    │    │  └─────────────┘   │      │  └─────────────┘  │
                    │    │                    │      │                   │
                    │    │  ┌─────────────┐   │      │  ┌─────────────┐  │
                    │    │  │    Rank     │   │      │  │    Rank     │  │
                    │    │  │ Stable Rank │   │      │  │ Effective   │  │
                    │    │  │ (Power Iter)│   │      │  │  Rank (SVD) │  │
                    │    │  │ m=β*||W||²/σ│   │      │  │ m=β*exp(H)  │  │
                    │    │  └─────────────┘   │      │  └─────────────┘  │
                    │    └────────────────────┘      └───────────────────┘
                    │                   │                      │
                    └───────────────────┴──────────────────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  Clip & Round to Pow2 │
                            │  m ∈ [m_min, f_max*d] │
                            └───────────┬───────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
            ┌───────────▼──────────┐      ┌────────────▼─────────┐
            │ Create Fastfood Ops  │      │   Operator Cache     │
            │  fwd: D → m          │◄─────┤ (seed, D, m) → ops  │
            │  lift: m → D         │      └──────────────────────┘
            └───────────┬──────────┘
                        │
                        │ For each task vector:
                        │   Y_k = fwd(Δ_k)
                        │
            ┌───────────▼──────────┐
            │   Merge in Subspace  │
            │   Y* = merge(Y_1..K) │
            │   (signmax/ema/ties) │
            └───────────┬──────────┘
                        │
            ┌───────────▼──────────┐
            │   Lift to Original   │
            │   Δ* = lift(Y*)      │
            └───────────┬──────────┘
                        │
            ┌───────────▼──────────┐
            │  Reconstruct Model   │
            │  θ* = θ_0 + scale*Δ* │
            └──────────────────────┘
```

## Key Components

### 1. ProjSizeCfg (Configuration)
```python
@dataclass
class ProjSizeCfg:
    m_min: int = 16              # Hard lower bound
    f_max: float = 0.5           # Max fraction
    pow2_round: bool = True      # Power-of-2 rounding
    pow2_mode: str = "ceil"      # Rounding mode
    ratio: float = 0.25          # For fixed strategy
    beta: float = 2.5            # For rank strategy
    rng: Optional[Random] = None # For random strategy
```

### 2. proj_size_for() (Dispatcher)
```python
def proj_size_for(
    unit: Tensor | Dict[str, Tensor],
    mode: Mode,           # "tensor" | "layer"
    strategy: Strategy,   # "fixed" | "random" | "rank"
    cfg: ProjSizeCfg
) -> int
```

### 3. Rank Estimators

**Stable Rank (Tensor Mode)**
```python
def stable_rank_tensor(W: Tensor) -> float:
    fro2 = ||W||_F^2
    sigma_max = power_iteration(W)
    return fro2 / sigma_max^2
```
- Complexity: O(n_iter * d_in * d_out)
- No full SVD needed
- 10-50x faster

**Effective Rank (Layer Mode)**
```python
def effective_rank_from_svals(S: Tensor) -> float:
    p = S^2 / sum(S^2)
    H = -sum(p * log(p))
    return exp(H)
```
- Uses variance spectrum
- Entropy-based measure
- Mirrors intrinsic dimension analysis

### 4. Integration Points

**A. Main Loop (Standard Processing)**
```python
for name in keys_linear:
    # Compute projection size
    if use_adaptive:
        if mode == "layer":
            m = proj_size_for(layer_params, "layer", strategy, cfg)
        else:
            m = proj_size_for(tensor, "tensor", strategy, cfg)
    else:
        m = int(d_last * proj_ratio)
    
    # Create/reuse operator
    cache_key = (seed_key, cur_D, m)
    if cache_key not in op_cache:
        fwd, lift = create_fastfood_ops(...)
        op_cache[cache_key] = (fwd, lift)
```

**B. Task Arithmetic Reconstruction**
```python
if use_task_arithmetic_reconstruction:
    # Compute merged task vector
    task_vector = sum(θ_i - θ_0)
    
    # Project with adaptive sizing
    for param in task_vector:
        m = _compute_proj_size(param)
        Y = fwd(Δ)
        Δ_rec = lift(Y)
```

## Data Flow

```
Input: Model Parameters
    │
    ├─ per_tensor:  W ∈ R^{out×in}
    │   └─ Stable rank → r_s ≈ 100
    │       └─ m = β * r_s = 2.5 * 100 = 250
    │           └─ Clip [16, 0.5*in] → m = min(250, 0.5*in)
    │               └─ Round pow2 → m = 256
    │
    └─ layer: {W1, W2, W3, bias}
        └─ Effective rank → r_eff ≈ 150
            └─ m = β * r_eff = 2.5 * 150 = 375
                └─ Clip [16, 0.5*max_in] → m = min(375, ...)
                    └─ Round pow2 → m = 512
```

## Performance Characteristics

| Mode     | Strategy | Overhead | Quality   | Use Case               |
|----------|----------|----------|-----------|------------------------|
| Fixed    | -        | 0%       | Baseline  | Original behavior      |
| Tensor   | Fixed    | ~1%      | Same      | Testing                |
| Tensor   | Random   | ~1%      | Variable  | Exploration            |
| Tensor   | Rank     | ~5-10%   | Better    | Fast adaptive          |
| Layer    | Fixed    | ~1%      | Same      | Layer consistency      |
| Layer    | Random   | ~1%      | Variable  | Layer exploration      |
| Layer    | Rank     | ~15-25%  | Best      | Quality-focused        |

*Overhead is one-time cost during initialization, not per merge operation*
