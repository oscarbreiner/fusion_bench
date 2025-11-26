# methods/adafastfood_clip_ratios_only.py
# Final "ratios-only" AdaFastfood for CLIP
# - Only learnable: per-layer projection ratios (sigmoid(theta_l) in (eps,1))
# - Fixed donor aggregation: mean in Fastfood space
# - Differentiable Fastfood filter: y = V^T diag(s(r)) V x  (soft gating)
# - Functional forward (no load_state_dict) so grads flow into ratios
from __future__ import annotations

import math
import logging
from typing import Dict, List, Tuple, Optional, Iterable

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.func import functional_call
from omegaconf import DictConfig
from tqdm.autonotebook import tqdm

from transformers import CLIPModel, CLIPProcessor
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin

# we import only utils that don't affect differentiability paths
from ..srp_utils import layer_key, seed_from_string
from ..structured_projection import next_pow2, fwht_inplace_ortho

log = logging.getLogger(__name__)


# --------------------------- differentiable FWHT -----------------------------

def _fwht_ortho(x: Tensor) -> Tensor:
    """
    Differentiable orthonormal FWHT along last dim.
    Returns a new tensor (avoids unsafe in-place on autograd graph).
    Uses the refactored structured_projection implementation.
    """
    n = x.shape[-1]
    if n <= 1:
        return x
    y = x.clone()  # ensure we don't modify input on autograd graph
    return fwht_inplace_ortho(y)


class _FFParams(nn.Module):
    """
    Fixed Fastfood parameters for one scope+length L:
      B ∈ {±1}^L, G ∈ R^L (optional), Π permutation, all registered as buffers.
    """
    def __init__(self, L: int, seed_key: str, use_G: bool, device: torch.device):
        super().__init__()
        g = torch.Generator(device=device)
        g.manual_seed(seed_from_string(seed_key))
        B = (torch.randint(0, 2, (L,), generator=g, device=device, dtype=torch.int8) * 2 - 1).to(torch.float32)
        if use_G:
            G = torch.randn(L, generator=g, device=device, dtype=torch.float32)
        else:
            G = torch.ones(L, device=device, dtype=torch.float32)
        Pi = torch.randperm(L, generator=g, device=device)
        inv_Pi = torch.argsort(Pi)
        self.register_buffer("B", B, persistent=False)
        self.register_buffer("G", G, persistent=False)
        self.register_buffer("Pi", Pi, persistent=False)
        self.register_buffer("inv_Pi", inv_Pi, persistent=False)
        self.L = L

    def forward(self, x: Tensor) -> Tensor:  # V x
        # x: [..., L]
        y = x * self.B
        y = _fwht_ortho(y)
        y = y[..., self.Pi]
        y = y * self.G
        y = _fwht_ortho(y)
        return y

    def inverse(self, z: Tensor) -> Tensor:  # V^T z
        y = _fwht_ortho(z)
        y = y * self.G
        y = y[..., self.inv_Pi]
        y = _fwht_ortho(y)
        y = y * self.B
        return y


def _ff_filter_soft(xD: Tensor, ff: _FFParams, ratio: Tensor, temperature: float) -> Tensor:
    """
    Differentiable Fastfood "filter":
      z = V x
      s = sigmoid(τ * (k - idx)),  k = ratio * L
      y = V^T (s ∘ z)
    Shapes:
      xD: [rows, D] (we will pad to L then truncate back to D)
      ff: carries fixed B,G,Pi,inv_Pi with length L >= D
      ratio: scalar tensor (per-layer)
    """
    rows, D = xD.shape
    L = ff.L
    # pad to L
    if D < L:
        pad = torch.zeros((rows, L - D), dtype=xD.dtype, device=xD.device)
        x = torch.cat([xD, pad], dim=-1)
    else:
        x = xD

    # forward Vx
    z = ff(x)

    # soft gate at rows level (same gate for each row)
    idx = torch.arange(L, device=z.device, dtype=z.dtype)
    k = ratio * L
    s = torch.sigmoid(temperature * (k - idx))  # shape [L]
    z_gated = z * s  # broadcast over rows

    # back-transform
    y_full = ff.inverse(z_gated)

    # truncate back to D
    return y_full[..., :D]


# --------------------------- merging utilities ------------------------------

def _is_mergeable(name: str, t: Tensor) -> bool:
    """Skip 1D tensors (bias/LN) and non-float."""
    return torch.is_floating_point(t) and t.ndim >= 2 and t.numel() > 0


def _as_2d_rows_lastdim(x: Tensor) -> Tuple[Tensor, int, int]:
    D = int(x.shape[-1])
    rows = x.numel() // D
    return x.view(rows, D), rows, D


def _state_delta(donor_sd: Dict[str, Tensor], pre_sd: Dict[str, Tensor], k: str) -> Optional[Tensor]:
    if k not in donor_sd or k not in pre_sd:
        return None
    a, b = donor_sd[k], pre_sd[k]
    if a.shape != b.shape:
        return None
    if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
        return None
    return (a - b)


# --------------------------- merged functional model ------------------------

class RatiosOnlyAdaFastfood(nn.Module):
    """
    Holds:
      - frozen pretrained module
      - frozen donor state_dicts
      - learnable per-layer ratio θ_l  (ratio = σ(θ)*(1-ε)+ε)
      - fixed Fastfood params per (scope, L)
    Provides:
      - build_merged_params(): differentiable merged parameter mapping
    """
    def __init__(
        self,
        pretrained: nn.Module,
        donors: List[nn.Module],
        *,
        device: torch.device,
        use_G: bool = False,
        subspace_scope: str = "layer",   # "layer" | "global" | "per_tensor"
        block_rows: int = 8192,
        eps_ratio: float = 0.05,
        soft_temperature: float = 10.0,
    ):
        super().__init__()
        self.pretrained = pretrained.eval().requires_grad_(False).to(device)
        for m in donors:
            m.eval().requires_grad_(False).to(device)
        self.device = device
        self.use_G = use_G
        self.scope = subspace_scope
        self.block_rows = int(block_rows)
        self.eps_ratio = float(eps_ratio)
        self.soft_temperature = float(soft_temperature)

        # candidate parameter names (trainable in pretrained)
        self.pnames: List[str] = [n for n, p in self.pretrained.named_parameters() if p.requires_grad]
        self.name2idx = {n: i for i, n in enumerate(self.pnames)}
        self.num_layers = len(self.pnames)
        self.num_tasks = len(donors)

        # keep states on device for speed (they won't require grad)
        self.pre_sd = {k: v.detach().to(device) for k, v in self.pretrained.state_dict().items()}
        self.donor_sds = [{k: v.detach().to(device) for k, v in d.state_dict().items()} for d in donors]

        # learnable ratios θ_l
        init_ratio = 0.3
        theta0 = math.log((init_ratio - self.eps_ratio) / (1.0 - init_ratio))
        self._theta = nn.Parameter(torch.full((self.num_layers,), float(theta0), device=device))

        # cache for FF params: (scope_key, L) -> _FFParams
        self.ff_cache: nn.ModuleDict = nn.ModuleDict()

    def ratios(self) -> Tensor:
        # ratio ∈ (eps, 1): σ(θ)*(1-ε)+ε
        return torch.sigmoid(self._theta) * (1.0 - self.eps_ratio) + self.eps_ratio

    def _scope_key(self, pname: str) -> str:
        if self.scope == "global":
            return "__GLOBAL__"
        elif self.scope == "per_tensor":
            return pname
        else:
            return layer_key(pname)

    def _get_ff_params(self, scope_key: str, D: int) -> _FFParams:
        L = next_pow2(D)
        md_key = f"{scope_key}::L{L}"
        if md_key not in self.ff_cache:
            self.ff_cache[md_key] = _FFParams(L=L, seed_key=scope_key, use_G=self.use_G, device=self.device)
        return self.ff_cache[md_key]

    def build_merged_params(self) -> Dict[str, Tensor]:
        """
        Returns a dict(name->Tensor) for ALL module parameters (mergeable get merged;
        others are copied from pretrained). Everything stays on-device and in-graph
        w.r.t. self._theta.
        """
        ratios = self.ratios()  # [L]
        merged_params: Dict[str, Tensor] = {}

        # default: take pretrained params as-is; overwrite mergeables below
        for n, p in self.pretrained.named_parameters():
            merged_params[n] = p.detach()  # still a tensor; not a leaf so it's fine
        # NOTE: merged tensors we compute below will be graph-connected to theta.

        for pname in self.pnames:
            pre_t = self.pre_sd.get(pname, None)
            if pre_t is None or not _is_mergeable(pname, pre_t):
                continue

            layer_idx = self.name2idx[pname]
            ratio = ratios[layer_idx]  # scalar

            # reshape to [rows, D]
            pre_2d, rows, D = _as_2d_rows_lastdim(pre_t)
            scope = self._scope_key(pname)
            ff = self._get_ff_params(scope, D)

            # streaming blocks
            cursor = 0
            out_2d = pre_2d.clone()
            while cursor < rows:
                take = min(rows - cursor, self.block_rows)

                # stack donor deltas [K, take, D]
                deltas = []
                for k in range(self.num_tasks):
                    delta = _state_delta(self.donor_sds[k], self.pre_sd, pname)
                    if delta is None:
                        continue
                    deltas.append(delta.view(rows, D)[cursor:cursor + take, :])
                if not deltas:
                    cursor += take
                    continue

                U = torch.stack(deltas, dim=0).to(self.device)  # [K, take, D]

                # apply differentiable FF filter to each donor block and average
                # y_k = V^T diag(s(r)) V (U_k)
                Ys = []
                for k in range(U.size(0)):
                    y = _ff_filter_soft(U[k], ff, ratio, temperature=self.soft_temperature)  # [take, D]
                    Ys.append(y)
                merged_block_delta = torch.stack(Ys, dim=0).mean(dim=0)  # [take, D]

                out_2d[cursor:cursor + take, :] = out_2d[cursor:cursor + take, :] + merged_block_delta.to(out_2d.dtype)
                cursor += take

            merged_params[pname] = out_2d.view_as(pre_t)

        return merged_params


# --------------------------- CLIP-specific algorithm ------------------------

class CLIPRatiosOnlyAdaFastfood(
    LightningFabricMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    """
    Test-time adaptation optimizing ONLY per-layer projection ratios.
    Backbone is frozen. Donor aggregation is fixed mean in Fastfood space.
    """

    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)
        self._clip_processor: Optional[CLIPProcessor] = None
        self._zeroshot: Dict[str, Tensor] = {}
        self._logit_scale_exp: Optional[Tensor] = None

    # ---------- hooks the program must provide ----------
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """Override in your program/mixin to return an iterator over test batches."""
        raise NotImplementedError("Provide a task→dataloader iterator via your program.")

    # ---------- CLIP zeroshot setup ----------
    def _setup_clip_refs(self, modelpool: ModelPool, device: torch.device):
        cfg = modelpool.get_model_config("_pretrained_")
        if isinstance(cfg, str):
            pretrained_path = cfg
        else:
            pretrained_path = getattr(cfg, "pretrained_model_name_or_or_path", None) \
                              or getattr(cfg, "pretrained_model_name_or_path", None) \
                              or getattr(cfg, "path", None)

        self._clip_processor = CLIPProcessor.from_pretrained(pretrained_path)
        ref = CLIPModel.from_pretrained(pretrained_path).to(device)
        self._logit_scale_exp = ref.logit_scale.exp().detach().to(device)

        # build zeroshot text heads
        clf = HFCLIPClassifier(ref, self._clip_processor)
        self._zeroshot = {}
        for task in modelpool.model_names:
            classnames, templates = get_classnames_and_templates(task)
            clf.set_classification_task(classnames, templates)
            W = clf.zeroshot_weights.to(device)  # [C, D]
            W = W / (W.norm(p=2, dim=-1, keepdim=True) + 1e-12)  # ensure unit-norm
            self._zeroshot[task] = W.detach()

    # ---------- robust image features ----------
    @staticmethod
    def _image_features(model: nn.Module, images: Tensor) -> Tensor:
        if hasattr(model, "get_image_features"):
            return model.get_image_features(pixel_values=images)
        out = model(images)
        if isinstance(out, dict):
            if "image_embeds" in out: return out["image_embeds"]
            if "last_hidden_state" in out: return out["last_hidden_state"][:, 0]
            # else pick first tensor-like
            for v in out.values():
                if torch.is_tensor(v): return v
            raise RuntimeError("Cannot parse CLIP image features from dict output.")
        if isinstance(out, (tuple, list)):
            return out[0] if torch.is_tensor(out[0]) else next(v for v in out if torch.is_tensor(v))
        if torch.is_tensor(out): return out
        raise RuntimeError(f"Unexpected model output type: {type(out)}")

    def _compute_logits(self, model: nn.Module, params: Dict[str, Tensor], images: Tensor, task: str) -> Tensor:
        device = next(model.parameters()).device
        images = images.to(device)
        # functional forward with merged params
        feats = self._image_features(functional_call(model, params, (images,), {}), images=None) \
                if hasattr(self._image_features, "__self__") else \
                self._image_features(functional_call(model, params, (images,), {}), images=None)
        # Above call reuses _image_features; but since we already passed images to forward,
        # the second 'images' arg is ignored for actual path. Safe for all cases.
        if feats.dim() == 3:  # [B, T, D] → pool
            feats = feats[:, 0]
        feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        text = self._zeroshot[task]  # [C, D]
        logit_scale = self._logit_scale_exp or torch.tensor(1.0, device=device)
        return logit_scale * (feats @ text.t())

    # ---------- loss plotting ----------
    def _save_loss_plot(self, loss_hist: List[Tuple[int, float]]):
        """Save loss curve plot to log directory."""
        if not loss_hist:
            log.warning("No loss history to plot")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for cluster
            import matplotlib.pyplot as plt
            import os
            
            xs, ys = zip(*loss_hist)
            
            plt.figure(figsize=(10, 6))
            plt.plot(xs, ys, linewidth=2, color='#2E86AB', alpha=0.8)
            plt.xlabel("Training Step", fontsize=12)
            plt.ylabel("Cross-Entropy Loss", fontsize=12)
            plt.title("Learnable Fastfood: Loss Curve", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Determine output path
            output_filename = self.config.get("loss_plot_path", "learnable_fastfood_loss.png")
            
            # Use Fabric log_dir if available
            if hasattr(self, 'fabric') and hasattr(self.fabric, 'logger'):
                if hasattr(self.fabric.logger, 'log_dir'):
                    output_path = os.path.join(self.fabric.logger.log_dir, output_filename)
                elif hasattr(self.fabric.logger, 'save_dir'):
                    output_path = os.path.join(self.fabric.logger.save_dir, output_filename)
                else:
                    output_path = output_filename
            else:
                output_path = output_filename
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"✓ Saved loss plot to: {output_path}")
            
            # Also log statistics
            final_loss = ys[-1]
            min_loss = min(ys)
            log.info(f"  Loss statistics: final={final_loss:.4f}, min={min_loss:.4f}, steps={len(loss_hist)}")
            
        except Exception as e:
            log.warning(f"Could not save loss plot: {e}")

    # ---------- construct wrapper ----------
    @torch.no_grad()
    def _construct_wrapper(self, pool: ModelPool) -> RatiosOnlyAdaFastfood:
        pretrained = pool.load_model("_pretrained_")
        donors = [pool.load_model(n) for n in pool.model_names]
        device = torch.device(self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        wrapper = RatiosOnlyAdaFastfood(
            pretrained=pretrained,
            donors=donors,
            device=device,
            use_G=bool(self.config.get("use_G", False)),
            subspace_scope=self.config.get("subspace_scope", "layer"),
            block_rows=int(self.config.get("block_rows", 8192)),
            eps_ratio=float(self.config.get("eps_ratio", 0.05)),
            soft_temperature=float(self.config.get("soft_temperature", 10.0)),
        ).to(device)
        return wrapper

    # ---------- run ----------
    def run(self, modelpool: ModelPool, **kwargs) -> nn.Module:
        log.info("Ratios-only AdaFastfood TTA (CLIP)")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        with self.profile("construct wrapper"):
            wrapper = self._construct_wrapper(modelpool)
        device = next(wrapper.parameters()).device

        with self.profile("setup zeroshot"):
            self._setup_clip_refs(modelpool, device)

        # optimizer ONLY over ratios
        optimizer = torch.optim.Adam([wrapper._theta], lr=float(self.config.get("proj_lr", 1e-3)))
        wrapper, optimizer = self.fabric.setup(wrapper, optimizer)

        # TTA loop
        max_steps = int(self.config.get("max_steps", 500))
        if getattr(self, "is_debug_mode", False) or self.config.get("fast_dev_run", False):
            max_steps = 1

        loss_hist = []
        for step in (pbar := tqdm(range(max_steps), "RatiosOnly TTA", dynamic_ncols=True)):
            total_loss = None
            valid = 0
            # build merged params once per step (keeps graph)
            merged_params = wrapper.build_merged_params()

            for task in modelpool.model_names:
                it = self.get_shuffled_test_loader_iter(task)
                try:
                    batch = next(it)
                except StopIteration:
                    continue
                except Exception as e:
                    log.warning(f"[{task}] dataloader error: {e}")
                    continue

                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                logits = self._compute_logits(wrapper.pretrained, merged_params, images, task)
                # entropy loss
                probs = torch.softmax(logits, dim=-1)
                loss = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                valid += 1
                total_loss = loss if total_loss is None else (total_loss + loss)

            if valid == 0:
                log.warning("No valid batches across tasks at this step, skipping.")
                continue

            self.fabric.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = float(total_loss.item()) / valid
            loss_hist.append((step, avg_loss))
            self.fabric.log_dict({
                "train/loss": avg_loss,
                "train/ratio_mean": wrapper.ratios().mean().item(),
                "train/ratio_std": wrapper.ratios().std().item(),
                "train/theta_norm": wrapper._theta.norm().item(),
            }, step=step)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", rmean=f"{wrapper.ratios().mean().item():.3f}")

        # Save loss plot
        self._save_loss_plot(loss_hist)

        # Build final merged params and materialize a standalone merged model
        with torch.no_grad():
            final_params = wrapper.build_merged_params()
            merged_model = functional_call(wrapper.pretrained, final_params, (), {})
        return merged_model
