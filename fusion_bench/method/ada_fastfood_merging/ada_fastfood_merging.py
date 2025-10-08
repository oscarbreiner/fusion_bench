"""
AdaFastFood Merging Algorithm

Hybrid approach combining AdaMerging with FastFood subspace projections:
1. Each layer learns optimal subspace projection ratio (learnable compression)
2. AdaMerging coefficients learned within compressed subspaces per layer per task
3. Efficient test-time adaptation with entropy loss
4. Uses FastFood operators with proper block processing and memory management

This implementation correctly follows the original FastFood block processing
and AdaMerging test-time adaptation patterns.
"""

import hashlib
import logging
import math
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool, to_modelpool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.utils.type import StateDictType

from ..adamerging.entropy_loss import entropy_loss
from ..adamerging.utils import get_memory_usage

log = logging.getLogger(__name__)

EPS = 1e-12

# ============================================================================
# FastFood Helper Functions (copied from original implementation)
# ============================================================================

def _next_pow2(n: int) -> int:
    """Find next power of 2 >= n"""
    return 1 << (n - 1).bit_length()

@torch.no_grad()
def _fwht_inplace_ortho(x: Tensor) -> Tensor:
    """In-place orthonormal FWHT along the last dim (scale 1/sqrt(n))."""
    n = x.shape[-1]
    if n <= 1:
        return x
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.view(-1, n)
        h *= 2
    x.mul_(1.0 / math.sqrt(n))
    return x

def _seed_from(s: str) -> int:
    """Generate deterministic seed from string"""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")

def _fastfood_ops(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
    use_G: bool,
):
    """
    Build a Fastfood operator with:
      V = H Π G H B ∈ R^{L×L}, L = 2^⌈log2 D⌉
      P = random row subset of size m = proj_dim
    We return:
      fwd(x)  = sqrt(L/m) * P V [x; 0]
      lift(y) = V^T P^T (y / sqrt(L/m))
    The same (B, G, Π, P) are reused for all tensors sharing `seed_key`.
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    L = _next_pow2(D)
    m = max(1, int(proj_dim))

    # Fastfood parameters
    B = (torch.randint(0, 2, (L,), dtype=torch.int8, device=device) * 2 - 1).to(
        dtype=torch.float32
    )
    G = (
        torch.randn(L, device=device, dtype=torch.float32)
        if use_G
        else torch.ones(L, device=device, dtype=torch.float32)
    )
    Pi = torch.randperm(L, device=device)
    inv_Pi = torch.argsort(Pi)

    # JL row subset and scaling (subsampled SRHT)
    row_idx = torch.randperm(L, device=device)[:m]
    scale = math.sqrt(L / m)

    def fwd(xD: Tensor) -> Tensor:
        assert xD.shape[-1] == D
        x = xD
        if D < L:
            x = torch.nn.functional.pad(x, (0, L - D))
        x = x.to(torch.float32, copy=False)
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)  # P V x
        return (scale * x).contiguous()

    def lift(y: Tensor) -> Tensor:
        y = (y.to(torch.float32, copy=False) / scale)
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)  # P^T y
        _fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        _fwht_inplace_ortho(y_full)
        y_full.mul_(B)  # V^T P^T y
        return y_full[..., :D].contiguous()

    return fwd, lift

def _layer_key(name: str) -> str:
    """Heuristic layer-grouping key (works for most HF models)."""
    parts = name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name

# ============================================================================
# AdaFastFood Wrapper Model
# ============================================================================

class AdaFastFoodMergedModel(nn.Module):
    """
    Hybrid AdaMerging + FastFood model that learns:
    1. Per-layer projection ratios (subspace dimensions)  
    2. Per-task per-layer AdaMerging coefficients within subspaces
    
    The model follows the FastFood block processing pattern and 
    AdaMerging dynamic weight updating.
    """
    
    _merged_state_dict: StateDictType = None
    
    def __init__(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        proj_init_strategy: str = "conservative",
        proj_init_value: float = 0.3,
        ada_init_value: float = None,
        use_G: bool = False,
        clamp_weights: bool = True,
        clamp_proj: bool = True,
        subspace_scope: str = "layer",
        block_rows: int = 8192,
        device: str = "cuda",
    ):
        """
        Initialize AdaFastFood merged model.
        
        Args:
            pretrained_model: Base pretrained model
            finetuned_models: List of fine-tuned models to merge
            proj_init_strategy: How to initialize projection ratios 
            proj_init_value: Initial projection ratio value
            ada_init_value: Initial AdaMerging coefficient
            use_G: Whether to use Gaussian scaling in FastFood
            clamp_weights: Whether to clamp AdaMerging weights
            clamp_proj: Whether to clamp projection ratios
            subspace_scope: "global" | "layer" | "per_tensor"
            block_rows: Block size for memory management
            device: Device for computations
        """
        super().__init__()
        
        self.num_tasks = len(finetuned_models)
        self.use_G = use_G
        self.clamp_weights = clamp_weights
        self.clamp_proj = clamp_proj
        self.subspace_scope = subspace_scope
        self.block_rows = block_rows
        self.device = torch.device(device)
        
        # Extract task vectors and layer information
        self._extract_task_vectors(pretrained_model, finetuned_models)
        
        # Initialize learnable parameters
        self._init_learnable_params(proj_init_strategy, proj_init_value, ada_init_value)
        
        # Store models
        self.pretrained_model = pretrained_model.requires_grad_(False)
        self.task_vectors = nn.ModuleList()
        for m in finetuned_models:
            m.requires_grad_(False)
            self.task_vectors.append(m)
            
    def _extract_task_vectors(self, pretrained_model: nn.Module, finetuned_models: List[nn.Module]):
        """Extract task vectors following AdaMerging pattern"""
        
        # Get trainable parameters (following AdaMerging pattern)
        trainable_params = []
        for name, param in pretrained_model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
                
        self.param_names = [name for name, _ in trainable_params]
        self.num_layers = len(trainable_params)
        
        print(f"AdaFastFood: Found {self.num_layers} trainable layers")
        
        # Convert fine-tuned models to task vectors (finetuned - pretrained)
        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                # Remove non-trainable parameters from fine-tuned models
                for m in finetuned_models:
                    if hasattr(m, name):
                        # Use del_attr equivalent
                        parts = name.split(".")
                        obj = m
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        if hasattr(obj, parts[-1]):
                            delattr(obj, parts[-1])
            else:
                # Convert to task vectors (difference from pretrained)
                for m in finetuned_models:
                    # Use get_attr equivalent
                    parts = name.split(".")
                    obj = m
                    for part in parts:
                        obj = getattr(obj, part)
                    obj.data = obj.data - param.data
                    
    def _init_learnable_params(self, proj_init_strategy: str, proj_init_value: float, ada_init_value: float):
        """Initialize learnable projection ratios and AdaMerging weights"""
        
        # Handle type conversion from Hydra configuration
        proj_init_value = float(proj_init_value)
        
        # Initialize projection ratios per layer
        if proj_init_strategy == "conservative":
            proj_init = torch.full((self.num_layers,), proj_init_value)
        elif proj_init_strategy == "layer_dependent":
            # Early layers more compressed, later layers less compressed
            proj_init = torch.linspace(0.2, 0.7, self.num_layers)
        elif proj_init_strategy == "uniform":
            proj_init = torch.full((self.num_layers,), 0.5)
        else:
            raise ValueError(f"Unknown proj_init_strategy: {proj_init_strategy}")
            
        self.proj_params = nn.Parameter(proj_init)
        
        # Initialize AdaMerging weights per task per layer
        # Handle string 'None' from Hydra configuration
        if ada_init_value is None or (isinstance(ada_init_value, str) and ada_init_value.lower() == 'none'):
            ada_init_value = 1.0 / self.num_tasks
            
        self.ada_weights = nn.Parameter(
            torch.full((self.num_tasks, self.num_layers), float(ada_init_value))
        )
        
        print(f"Initialized proj_params: {self.proj_params.shape}")
        print(f"Initialized ada_weights: {self.ada_weights.shape}")
        
    def _clamp_parameters(self):
        """Clamp parameters to valid ranges"""
        with torch.no_grad():
            if self.clamp_proj:
                self.proj_params.clamp_(0.1, 1.0)
            if self.clamp_weights:
                self.ada_weights.clamp_(0.0, 2.0)
    
    def merge_weights(self):
        """
        Perform hybrid AdaFastFood merging following FastFood block processing pattern.
        This is the core fusion logic that combines FastFood projection with AdaMerging.
        """
        # Clamp parameters to valid ranges
        self._clamp_parameters()
        
        # Start with pretrained model state (following AdaMerging pattern)
        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        
        # Get all eligible parameters (float tensors that exist in all models)
        keys_float = []
        for name in self.param_names:
            param = state_dict[name]
            if torch.is_floating_point(param) and param.ndim >= 1:
                # Check if parameter exists in all task vectors
                all_exist = True
                for task_vector in self.task_vectors:
                    if not hasattr(task_vector, name):
                        all_exist = False
                        break
                if all_exist:
                    keys_float.append(name)
        
        print(f"Processing {len(keys_float)} float parameters")
        
        # FastFood operator cache (following original pattern)
        op_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}
        
        # Determine global dimension if using global scope
        global_D = None
        if self.subspace_scope == "global":
            max_d = 1
            for name in keys_float:
                param = state_dict[name] 
                max_d = max(max_d, int(param.shape[-1]))
            global_D = max_d
            
        def proj_seed_key(param_name: str) -> str:
            """Generate seed key based on subspace scope"""
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            elif self.subspace_scope == "layer":
                return _layer_key(param_name)
            else:  # per_tensor
                return param_name
        
        # Process each parameter (following FastFood block processing)
        merged_tensors = 0
        
        for name in keys_float:
            # Find the correct layer index in param_names
            try:
                layer_idx = self.param_names.index(name)
            except ValueError:
                print(f"Warning: parameter {name} not found in param_names, skipping")
                continue
                
            param = state_dict[name]
            d_last = int(param.shape[-1])
            rows = param.numel() // d_last
            
            if rows <= 0:
                continue
                
            # Convert to 2D view for FastFood processing
            param_2d = param.view(rows, d_last).float()
            block_rows = min(self.block_rows, rows)
            
            # Get projection parameters for this layer
            proj_ratio = self.proj_params[layer_idx]
            ada_layer_weights = self.ada_weights[:, layer_idx]  # [num_tasks]
            
            # Build FastFood operators (following original caching pattern)
            seed_key = proj_seed_key(name)
            cur_D = global_D if global_D is not None else d_last
            proj_dim = max(1, int(proj_ratio.item() * cur_D))
            
            cache_key = (seed_key, cur_D, proj_dim)
            if cache_key not in op_cache:
                fwd, lift = _fastfood_ops(
                    cur_D, proj_dim, seed_key=seed_key, device=self.device, use_G=self.use_G
                )
                op_cache[cache_key] = (fwd, lift)
            else:
                fwd, lift = op_cache[cache_key]
            
            # Process in blocks (following original FastFood pattern)
            cursor = 0
            param_changed = False
            
            while cursor < rows:
                take = min(rows - cursor, block_rows)
                param_block = param_2d[cursor:cursor + take, :]
                
                # Collect task vectors for this block
                task_deltas = []
                for task_idx in range(self.num_tasks):
                    task_vector = self.task_vectors[task_idx]
                    # Get task parameter
                    parts = name.split(".")
                    task_param = task_vector
                    for part in parts:
                        task_param = getattr(task_param, part)
                    
                    # Extract block and compute delta
                    task_param_2d = task_param.view(rows, d_last).float()
                    task_block = task_param_2d[cursor:cursor + take, :]
                    
                    # Pad if using global scope
                    if global_D is not None and d_last < cur_D:
                        padded_block = torch.zeros((take, cur_D), dtype=torch.float32, device="cpu")
                        padded_block[:, :d_last] = task_block
                        task_deltas.append(padded_block)
                    else:
                        task_deltas.append(task_block)
                
                # Project all task vectors to subspace
                projected_deltas = []
                for delta in task_deltas:
                    projected = fwd(delta.to(self.device, non_blocking=True))
                    projected_deltas.append(projected)
                
                # Apply AdaMerging coefficients in subspace
                merged_subspace = torch.zeros_like(projected_deltas[0])
                for task_idx, (weight, proj_delta) in enumerate(zip(ada_layer_weights, projected_deltas)):
                    merged_subspace += weight * proj_delta
                
                # Lift back to original space
                merged_full = lift(merged_subspace).to("cpu", non_blocking=True)
                merged_delta = merged_full[:, :d_last]
                
                # Update parameter block
                update = merged_delta.to(param_block.dtype)
                param_block.add_(update)
                
                # Track changes
                param_changed = param_changed or bool(update.abs().max().item() > 0)
                
                cursor += take
                
            merged_tensors += 1
            
            # Update state dict with modified parameter
            state_dict[name] = param_2d.view(param.shape).to(param.dtype)
        
        self._merged_state_dict = state_dict
        print(f"AdaFastFood merged {merged_tensors} parameters")
        return state_dict
    
    def merge_and_unload(self):
        """Merge weights and return standalone model"""
        self.merge_weights()
        self.pretrained_model.load_state_dict(self._merged_state_dict)
        return self.pretrained_model
        
    def forward(self, *args, **kwargs):
        """Forward pass through merged model"""
        if self._merged_state_dict is None:
            self.merge_weights()
        
        # Use functional call like AdaMerging
        from torch.func import functional_call
        return functional_call(
            self.pretrained_model, self._merged_state_dict, args, kwargs
        )
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics for analysis"""
        stats = {}
        total_params = 0
        total_compressed = 0
        
        for layer_idx in range(self.num_layers):
            proj_ratio = self.proj_params[layer_idx].item()
            total_params += 1
            total_compressed += proj_ratio
            
            if layer_idx < 5:  # Show first 5 layers
                stats[f"layer_{layer_idx}_ratio"] = proj_ratio
        
        stats["avg_compression_ratio"] = total_compressed / max(1, total_params)
        stats["memory_savings"] = 1 - (total_compressed / max(1, total_params))
        
        return stats

# ============================================================================
# Main Algorithm Class  
# ============================================================================

class AdaFastFoodMergingAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    """
    AdaFastFood Merging Algorithm
    
    Hybrid approach that combines:
    - FastFood subspace projections for computational efficiency
    - AdaMerging adaptive coefficients for task-specific control
    - Per-layer learnable compression ratios
    - Test-time adaptation via entropy loss
    """
    
    def __init__(self, algorithm_config: DictConfig):
        """Initialize algorithm with configuration"""
        super().__init__(algorithm_config)
        
    @torch.no_grad()
    def construct_ada_fastfood_merged_model(
        self, modelpool: ModelPool
    ) -> AdaFastFoodMergedModel:
        """Construct hybrid model from model pool"""
        
        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        
        # Create hybrid model 
        module = AdaFastFoodMergedModel(
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            proj_init_strategy=self.config.get("proj_init_strategy", "conservative"),
            proj_init_value=self.config.get("proj_init_value", 0.3),
            ada_init_value=self.config.get("ada_init_value", None),
            use_G=self.config.get("use_G", False),
            clamp_weights=self.config.get("clamp_weights", True),
            clamp_proj=self.config.get("clamp_proj", True),
            subspace_scope=self.config.get("subspace_scope", "layer"),
            block_rows=self.config.get("block_rows", 8192),
            device=self.config.get("device", "cuda"),
        )
        
        print(f"AdaFastFood model created with {module.num_tasks} tasks, {module.num_layers} layers")
        return module
        
    @rank_zero_only
    def save_merging_weights(self, file_path: str, module: AdaFastFoodMergedModel):
        """Save learned parameters"""
        import os
        
        if self.fabric.is_global_zero and self.config.get("save_merging_weights", False):
            save_path = file_path
            if not file_path.startswith(("/", ".")):
                save_path = os.path.join(self.log_dir, file_path)
                
            log.info(f"Saving AdaFastFood weights to {save_path}")
            
            weights_dict = {
                "proj_params": module.proj_params.detach().cpu(),
                "ada_weights": module.ada_weights.detach().cpu(),
                "compression_stats": module.get_compression_stats(),
                "config": {
                    "num_tasks": module.num_tasks,
                    "num_layers": module.num_layers,
                    "subspace_scope": module.subspace_scope,
                }
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(weights_dict, save_path)
    
    def run(self, modelpool: ModelPool, **kwargs) -> nn.Module:
        """Run the AdaFastFood merging algorithm"""
        
        log.info("Fusing models using AdaFastFood (AdaMerging + FastFood)")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)
        
        with self.profile("construct model"):
            module = self.construct_ada_fastfood_merged_model(modelpool)
            
        # Check if weights provided (skip adaptation)
        if self.config.get("weights", None) is not None:
            log.info("Pre-trained weights provided, skipping test-time adaptation")
            return module.merge_and_unload()
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
                
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module
                )
                
            stats = module.get_compression_stats()
            log.info(f"Final compression stats: {stats}")
            
            return module.merge_and_unload()
    
    def on_test_time_adaptation_start(self):
        """Setup before test-time adaptation"""
        pass
        
    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """Get test data loader (to be implemented by subclass)"""
        pass
        
    @abstractmethod
    def compute_logits(self, module: AdaFastFoodMergedModel, batch: Tensor, task: str) -> Tensor:
        """Compute logits (to be implemented by subclass)"""
        pass
    
    def test_time_adaptation(self, module: AdaFastFoodMergedModel) -> AdaFastFoodMergedModel:
        """
        Perform test-time adaptation following AdaMerging pattern.
        Jointly optimizes projection ratios and AdaMerging coefficients.
        """
        self.on_test_time_adaptation_start()
        
        # Check if any test data is available
        has_test_data = False
        for task in self.modelpool.model_names:
            try:
                data_iter = self.get_shuffled_test_loader_iter(task)
                next(data_iter)
                has_test_data = True
                break
            except StopIteration:
                continue
            except Exception:
                continue
        
        if not has_test_data:
            log.warning("No test data available for any task, skipping test-time adaptation")
            return module
        
        # Configure optimizer for both parameter types
        if self.config.get("optimizer", "adam") == "adam":
            optimizer = torch.optim.Adam([
                {
                    "params": [module.proj_params],
                    "lr": self.config.get("proj_lr", self.config.get("lr", 1e-3)),
                },
                {
                    "params": [module.ada_weights],
                    "lr": self.config.get("ada_lr", self.config.get("lr", 1e-3)),
                }
            ])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.get('optimizer')}")
        
        module, optimizer = self.fabric.setup(module, optimizer)
        
        # Training loop (following AdaMerging pattern)
        module.train()
        module.merge_weights()  # Initial merge
        
        max_steps = self.config.get("max_steps", 1000)
        if self.is_debug_mode:
            max_steps = 1
            
        for step_idx in (
            pbar := tqdm(
                range(max_steps),
                ("[DEBUG MODE] " if self.is_debug_mode else "") + "AdaFastFood Test-time Adaptation",
                dynamic_ncols=True,
            )
        ):
            total_loss = 0.0
            valid_tasks = 0
            
            # Iterate through all tasks (following AdaMerging pattern)
            for task in self.modelpool.model_names:
                try:
                    data_iter = self.get_shuffled_test_loader_iter(task)
                    batch = next(data_iter)
                    
                    if batch is None or len(batch) == 0:
                        log.warning(f"Empty batch for task {task}, skipping")
                        continue
                except StopIteration:
                    log.warning(f"No test data available for task {task}, skipping")
                    continue
                except Exception as e:
                    log.error(f"Error getting test data for task {task}: {e}, skipping")
                    continue
                    
                logits = self.compute_logits(module, batch[0], task)
                loss = entropy_loss(logits)
                total_loss += loss.item()
                valid_tasks += 1
                
                self.fabric.backward(loss, retain_graph=True)
            
            # Skip optimizer step if no valid data was processed
            if valid_tasks == 0:
                log.warning(f"No valid test data for any task in step {step_idx}, skipping optimization")
                continue
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Recompute merged weights with new parameters
            module.merge_weights()
            
            # Logging
            avg_loss = total_loss / valid_tasks
            compression_stats = module.get_compression_stats()
            
            metrics = {
                "train/loss": avg_loss,
                "train/proj_mean": module.proj_params.mean().item(),
                "train/proj_std": module.proj_params.std().item(),
                "train/ada_mean": module.ada_weights.mean().item(),
                "train/compression": compression_stats["avg_compression_ratio"],
            }
            
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "compression": f"{compression_stats['avg_compression_ratio']:.3f}"
            })
        
        # Log final learned parameters
        self._log_final_learned_parameters(module)
        
        # Create visualization if requested
        if self.config.get("plot_learned_params", True):
            self._plot_learned_parameters(module)
        
        log.info(get_memory_usage("After AdaFastFood adaptation"))
        self.print_profile_summary()
        
        return module
    
    def _log_final_learned_parameters(self, module: AdaFastFoodMergedModel):
        """Log detailed information about final learned parameters"""
        log.info("=" * 60)
        log.info("FINAL LEARNED PARAMETERS - AdaFastFood Training Complete")
        log.info("=" * 60)
        
        # Projection ratios analysis
        proj_ratios = module.proj_params.detach().cpu().numpy()
        log.info(f"📊 PROJECTION RATIOS (Compression Levels):")
        log.info(f"   Mean: {proj_ratios.mean():.3f} ± {proj_ratios.std():.3f}")
        log.info(f"   Range: [{proj_ratios.min():.3f}, {proj_ratios.max():.3f}]")
        log.info(f"   Median: {np.median(proj_ratios):.3f}")
        
        # Per-layer projection analysis
        for i, ratio in enumerate(proj_ratios[:10]):  # Show first 10 layers
            log.info(f"   Layer {i:2d}: {ratio:.3f} ({ratio*100:.1f}% compression)")
        if len(proj_ratios) > 10:
            log.info(f"   ... and {len(proj_ratios)-10} more layers")
            
        # AdaMerging weights analysis
        ada_weights = module.ada_weights.detach().cpu().numpy()
        log.info(f"\n🎯 ADAMIERGING TASK WEIGHTS:")
        log.info(f"   Shape: {ada_weights.shape} (tasks × layers)")
        log.info(f"   Overall mean: {ada_weights.mean():.3f} ± {ada_weights.std():.3f}")
        
        # Per-task analysis
        task_names = getattr(self.modelpool, 'model_names', [f"Task_{i}" for i in range(module.num_tasks)])
        for task_idx, task_name in enumerate(task_names):
            task_weights = ada_weights[task_idx]
            log.info(f"   {task_name}: mean={task_weights.mean():.3f}, std={task_weights.std():.3f}")
            
        # Compression statistics
        stats = module.get_compression_stats()
        log.info(f"\n💾 COMPRESSION STATISTICS:")
        log.info(f"   Average compression ratio: {stats['avg_compression_ratio']:.3f}")
        log.info(f"   Memory savings: {stats['memory_savings']*100:.1f}%")
        
        # Layer-wise dominance analysis
        log.info(f"\n🏆 TASK DOMINANCE ANALYSIS:")
        dominant_tasks = np.argmax(ada_weights, axis=0)
        for task_idx in range(module.num_tasks):
            count = np.sum(dominant_tasks == task_idx)
            task_name = task_names[task_idx] if task_idx < len(task_names) else f"Task_{task_idx}"
            log.info(f"   {task_name} dominates {count}/{module.num_layers} layers ({count/module.num_layers*100:.1f}%)")
            
        log.info("=" * 60)
    
    def _plot_learned_parameters(self, module: AdaFastFoodMergedModel):
        """Create visualization of learned parameters"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
            
            # Set style for cool plots
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Get data
            proj_ratios = module.proj_params.detach().cpu().numpy()
            ada_weights = module.ada_weights.detach().cpu().numpy()
            task_names = getattr(self.modelpool, 'model_names', [f"Task_{i}" for i in range(module.num_tasks)])
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12), facecolor='#1e1e1e')
            fig.suptitle('AdaFastFood Learned Parameters', fontsize=20, color='white', y=0.95)
            
            # 1. Projection Ratios Over Layers
            ax1 = plt.subplot(2, 3, 1)
            layers = np.arange(len(proj_ratios))
            plt.plot(layers, proj_ratios, 'o-', color='#00ff88', linewidth=2, markersize=4)
            plt.axhline(y=proj_ratios.mean(), color='#ff6b6b', linestyle='--', alpha=0.8, label=f'Mean: {proj_ratios.mean():.3f}')
            plt.fill_between(layers, proj_ratios.mean() - proj_ratios.std(), proj_ratios.mean() + proj_ratios.std(), 
                           alpha=0.2, color='#ff6b6b')
            plt.xlabel('Layer Index', color='white')
            plt.ylabel('Projection Ratio', color='white')
            plt.title('Learned Compression per Layer', color='white', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            ax1.tick_params(colors='white')
            
            # 2. Projection Ratio Distribution
            ax2 = plt.subplot(2, 3, 2)
            plt.hist(proj_ratios, bins=20, color='#4ecdc4', alpha=0.8, edgecolor='white')
            plt.axvline(proj_ratios.mean(), color='#ff6b6b', linestyle='--', linewidth=2, label=f'Mean: {proj_ratios.mean():.3f}')
            plt.xlabel('Projection Ratio', color='white')
            plt.ylabel('Frequency', color='white') 
            plt.title('Compression Distribution', color='white', fontweight='bold')
            plt.legend()
            ax2.tick_params(colors='white')
            
            # 3. AdaMerging Weights Heatmap
            ax3 = plt.subplot(2, 3, 3)
            im = plt.imshow(ada_weights, aspect='auto', cmap='plasma', interpolation='nearest')
            plt.colorbar(im, ax=ax3, label='Weight Value')
            plt.xlabel('Layer Index', color='white')
            plt.ylabel('Task Index', color='white')
            plt.title('Task-Layer Weight Matrix', color='white', fontweight='bold')
            
            # Add task names if available
            if len(task_names) <= 10:  # Only show names if not too many tasks
                plt.yticks(range(len(task_names)), [name[:15] for name in task_names])
            ax3.tick_params(colors='white')
            
            # 4. Task Weight Statistics
            ax4 = plt.subplot(2, 3, 4)
            task_means = ada_weights.mean(axis=1)
            task_stds = ada_weights.std(axis=1)
            x_pos = np.arange(len(task_names))
            
            bars = plt.bar(x_pos, task_means, yerr=task_stds, capsize=5, 
                          color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'][:len(task_names)],
                          alpha=0.8, edgecolor='white')
            plt.xlabel('Tasks', color='white')
            plt.ylabel('Average Weight', color='white')
            plt.title('Average Task Weights', color='white', fontweight='bold')
            
            # Rotate labels if many tasks
            if len(task_names) > 5:
                plt.xticks(x_pos, [name[:10] for name in task_names], rotation=45, ha='right')
            else:
                plt.xticks(x_pos, task_names)
            ax4.tick_params(colors='white')
            
            # 5. Layer-wise Task Dominance
            ax5 = plt.subplot(2, 3, 5)
            dominant_tasks = np.argmax(ada_weights, axis=0)
            dominance_counts = [np.sum(dominant_tasks == i) for i in range(module.num_tasks)]
            
            colors = plt.cm.Set3(np.linspace(0, 1, module.num_tasks))
            wedges, texts, autotexts = plt.pie(dominance_counts, labels=[f'{name}\n({count} layers)' for name, count in zip(task_names, dominance_counts)], 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Layer Dominance by Task', color='white', fontweight='bold')
            
            # 6. Compression vs Performance Scatter (if we had performance data)
            ax6 = plt.subplot(2, 3, 6)
            # Create a synthetic "complexity" measure based on weights
            layer_complexity = ada_weights.std(axis=0)  # How much tasks vary per layer
            plt.scatter(proj_ratios, layer_complexity, c=range(len(proj_ratios)), 
                       cmap='viridis', s=60, alpha=0.8, edgecolors='white')
            plt.colorbar(label='Layer Index')
            plt.xlabel('Projection Ratio (Compression)', color='white')
            plt.ylabel('Task Weight Variance', color='white')
            plt.title('Compression vs Task Complexity', color='white', fontweight='bold')
            ax6.tick_params(colors='white')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save plot
            save_path = self.config.get("plot_save_path", "ada_fastfood_learned_params.png")
            if not save_path.startswith(("/", ".")):
                import os
                save_path = os.path.join(getattr(self, 'log_dir', 'outputs'), save_path)
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
            log.info(f"📈 Saved parameter visualization to: {save_path}")
            
            # Also save interactive version if possible
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create interactive plotly version
                plotly_fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Projection Ratios', 'AdaMerging Weights Heatmap', 
                                  'Task Averages', 'Layer Dominance'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"type": "domain"}]]
                )
                
                # Add traces...
                plotly_fig.add_trace(go.Scatter(y=proj_ratios, mode='lines+markers', 
                                              name='Projection Ratio'), row=1, col=1)
                
                interactive_path = save_path.replace('.png', '_interactive.html')
                plotly_fig.write_html(interactive_path)
                log.info(f"📊 Saved interactive visualization to: {interactive_path}")
                
            except ImportError:
                log.info("📊 Plotly not available, skipping interactive plot")
                
            plt.show()
            
        except ImportError as e:
            log.warning(f"Could not create visualization: {e}. Install matplotlib and seaborn for plotting.")
        except Exception as e:
            log.error(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()