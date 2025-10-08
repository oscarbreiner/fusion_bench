from __future__ import annotations
import math, hashlib, re
from typing import Any, Dict, List, Tuple

import torch
from torch import nn, Tensor

from fusion_bench import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

EPS = 1e-12

# ---------------- Fastfood operators ----------------
def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

@torch.no_grad()
def _fwht_inplace_ortho(x: Tensor) -> Tensor:
    n = x.shape[-1]
    if n <= 1: return x
    h = 1
    while h < n:
        x = x.view(-1, n // (2*h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.view(-1, n)
        h *= 2
    x.mul_(1.0 / math.sqrt(n))
    return x

def _fastfood_ops(global_dim: int, proj_dim: int, seed: int, device: torch.device, use_G: bool):
    torch.manual_seed(seed)
    D = int(global_dim)
    L = _next_pow2(D)
    B  = (torch.randint(0, 2, (L,), dtype=torch.int8, device=device) * 2 - 1).to(dtype=torch.float32)
    G  = torch.randn(L, device=device, dtype=torch.float32) if use_G else torch.ones(L, device=device, dtype=torch.float32)
    Pi = torch.randperm(L, device=device)
    inv_Pi = torch.argsort(Pi)

    def fwd(xD: Tensor) -> Tensor:
        assert xD.shape[-1] == D
        x = xD
        if D < L: x = torch.nn.functional.pad(x, (0, L - D))
        x = x.to(torch.float32, copy=False)
        x.mul_(B); _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G); _fwht_inplace_ortho(x)
        return x[..., :proj_dim].contiguous()      # P · V · x

    def lift(y: Tensor) -> Tensor:
        m = y.shape[-1]
        y_pad = y
        if m < L: y_pad = torch.nn.functional.pad(y, (0, L - m))  # P^T
        y_pad = y_pad.to(torch.float32, copy=False)
        _fwht_inplace_ortho(y_pad); y_pad.mul_(G)
        y_pad = y_pad[..., inv_Pi]; _fwht_inplace_ortho(y_pad)
        y_pad.mul_(B)
        return y_pad[..., :D].contiguous()         # V^T · P^T · y

    return fwd, lift

def _seed_from(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:4], "little")

# --------------- Alignment / masks / α solving / rescale ---------------
@torch.no_grad()
def _ties_sign_and_trim(U: Tensor, trim_pct: float = 0.0):
    # Elect global sign per column by sum, trim per-donor magnitudes if requested
    s = torch.sign(U.sum(dim=0))
    s[s == 0] = 1.0
    U = U * s  # broadcast
    if trim_pct and trim_pct > 0:
        q = max(0.0, min(100.0, float(trim_pct)))
        thr = torch.quantile(U.abs().reshape(-1), q / 100.0)
        U = U * (U.abs() >= thr)
    return U

@torch.no_grad()
def _tadrop_mask(U: Tensor, tau: float = 0.5, eps: float = EPS):
    # Keep entries that agree across donors within relative diff tau
    # U: [K, N, m] donors x rows x dims (or [K, m] if reduced)
    ref = U.mean(dim=0, keepdim=True)
    denom = U.abs() + ref.abs() + eps
    d = (U - ref).abs() / denom
    m = (d <= tau).to(U.dtype)
    return U * m

@torch.no_grad()
def _pareto_weights(Ys: List[Tensor]) -> List[float]:
    # 2-donor closed form; for K>2 use uniform (could extend to NNLS later)
    K = len(Ys)
    if K != 2:
        return [1.0 / K] * K
    U, V = Ys[0], Ys[1]
    D = (V - U)
    num = (D * V).sum()
    den = (D * D).sum() + EPS
    a = torch.clamp(num / den, 0.0, 1.0).item()
    return [1.0 - a, a]

def _layer_key(name: str) -> str:
    # crude grouping: everything up to the last two components (works for CLIP/Transformers)
    parts = name.split(".")
    if len(parts) >= 3: return ".".join(parts[:3])
    if len(parts) >= 2: return ".".join(parts[:2])
    return name

# ------------------ The Algorithm ------------------
@auto_register_config
class FastfoodSubspaceMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    General subspace merge with Fastfood operators.

    Options:
      - subspace_scope: "per_tensor" (default) | "layer" | "global"
      - merge_where: "subspace" (default) | "postlift"
      - align_mode: "none" | "ties" | "tadrop"
      - ties_trim_pct, tadrop_tau
      - use_pareto: bool (supports 2 donors now)
      - use_rescale: bool (EMR-R style)
      - proj_ratio, use_G, block_rows, weights[], scale
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",         # per_tensor | layer | global
        merge_where: str = "subspace",              # subspace | postlift
        align_mode: str = "none",                   # none | ties | tadrop
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.5,
        use_pareto: bool = False,
        use_rescale: bool = False,
        block_rows: int = 8192,                     # chunking rows per tensor
        weights: List[float] | None = None,         # donor weights (uniform if None)
        scale: float = 1.0,                         # post-merge scale on Δ*
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio   = float(proj_ratio)
        self.use_G        = bool(use_G)
        self.device       = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        self.merge_where  = str(merge_where)
        self.align_mode   = str(align_mode)
        self.ties_trim_pct= float(ties_trim_pct)
        self.tadrop_tau   = float(tadrop_tau)
        self.use_pareto   = bool(use_pareto)
        self.use_rescale  = bool(use_rescale)
        self.block_rows   = int(block_rows)
        self.weights      = list(weights) if weights is not None else None
        self.scale        = float(scale)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any) -> nn.Module:
        modelpool = to_modelpool(modelpool)

        with self.profile("loading models"):
            base_model = modelpool.load_model("_pretrained_")
            donor_names = list(modelpool.model_names)
            if len(donor_names) < 2:
                raise ValueError(f"Need ≥2 donors; got {len(donor_names)}")
            donors_sd: List[StateDictType] = [modelpool.load_model(n).state_dict(keep_vars=True) for n in donor_names]
            base_sd: Dict[str, Tensor] = base_model.state_dict(keep_vars=True)

        # weights
        if self.weights is None:
            w = [1.0 / len(donor_names)] * len(donor_names)
        else:
            if len(self.weights) != len(donor_names):
                raise ValueError("`weights` length must match number of donors.")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]

        # choose projection seed key per tensor / layer / global
        def proj_seed_key(param_name: str) -> str:
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            if self.subspace_scope == "layer":
                return _layer_key(param_name)
            return param_name  # per_tensor

        # Determine global D for "global" scope (max last-dim across all tensors)
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for name, t in base_sd.items():
                if torch.is_floating_point(t) and t.ndim >= 1:
                    maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd

        # Work on CPU copies (safer memory)
        base_cpu = {k: v.detach().cpu() for k, v in base_sd.items()}
        donors_cpu = [{k: v.detach().cpu() for k, v in d.items()} for d in donors_sd]
        dev = self.device
        merged = 0

        with self.profile("merging models"):
            # cache of operators keyed by seed_key and d_last (different d require different ops)
            op_cache: Dict[Tuple[str,int,int], Tuple[Any,Any]] = {}

            for name, tb in base_cpu.items():
                if not torch.is_floating_point(tb) or tb.ndim < 1:
                    continue
                shape = tb.shape
                if not all((name in d) and torch.is_floating_point(d[name]) and d[name].shape == shape for d in donors_cpu):
                    continue

                d_last = int(shape[-1])
                rows = tb.numel() // d_last
                if rows == 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)
                seed_key = proj_seed_key(name)
                # for global scope, project to proj_ratio * global_D; else proj_ratio * d_last
                cur_D = global_D if (global_D is not None) else d_last
                proj_dim = max(1, int(cur_D * self.proj_ratio))
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = _fastfood_ops(cur_D, proj_dim, seed=_seed_from(seed_key), device=dev, use_G=self.use_G)
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                # stream over rows
                cursor = 0
                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor+take, :]

                    # Prepare donor deltas buffers aligned to cur_D (for global scope we pad)
                    Xs = []
                    for dsd in donors_cpu:
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor:cursor+take, :]
                        delta = sl_donor - sl_base
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros((take, cur_D), dtype=torch.float32)
                            buf[:, :d_last].copy_(delta)
                            Xs.append(buf)
                        else:
                            Xs.append(delta)

                    # Project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # Optionally change mixing space
                    if self.merge_where == "postlift":
                        # reconstruct to original space before mixing
                        Xhats = [lift(Y).to("cpu", non_blocking=True)[:, :d_last] for Y in Ys]
                        mix_space = "postlift"
                    else:
                        mix_space = "subspace"

                    # Align / mask
                    if self.align_mode != "none":
                        if mix_space == "subspace":
                            U = torch.stack(Ys, dim=0)  # [K, take, m]
                        else:
                            U = torch.stack(Xhats, dim=0)  # [K, take, d]
                        if self.align_mode == "ties":
                            # sign-elect + (optional) trim on stacked donors
                            U = _ties_sign_and_trim(U, trim_pct=self.ties_trim_pct)
                        elif self.align_mode == "tadrop":
                            U = _tadrop_mask(U, tau=self.tadrop_tau)
                        # unpack back
                        if mix_space == "subspace":
                            Ys = [U[i] for i in range(U.shape[0])]
                        else:
                            Xhats = [U[i] for i in range(U.shape[0])]

                    # Weights (optionally Pareto for K=2 in mixing space)
                    if self.use_pareto and len(Ys) == 2:
                        cur_w = _pareto_weights(Ys if mix_space == "subspace" else Xhats)
                    else:
                        cur_w = w

                    # Merge in chosen space
                    if mix_space == "subspace":
                        Ymerge = torch.zeros_like(Ys[0])
                        for wi, Yi in zip(cur_w, Ys):
                            Ymerge.add_(wi * Yi)
                        Xmerge = lift(Ymerge).to("cpu", non_blocking=True)
                        Xmerge = Xmerge[:, :d_last]
                        # EMR-R rescale (match magnitude)
                        if self.use_rescale:
                            num = 0.0
                            for wi, Xorig in zip(cur_w, Xs):
                                num += wi * float(torch.linalg.vector_norm(Xorig))
                            den = float(torch.linalg.vector_norm(Xmerge)) + EPS
                            lam = num / den if den > 0 else 1.0
                            Xmerge = lam * Xmerge
                    else:
                        Xmerge = torch.zeros_like(Xhats[0])
                        for wi, Xi in zip(cur_w, Xhats):
                            Xmerge.add_(wi * Xi)
                        if self.use_rescale:
                            num = 0.0
                            for wi, Xi in zip(cur_w, Xhats):
                                num += wi * float(torch.linalg.vector_norm(Xi))
                            den = float(torch.linalg.vector_norm(Xmerge)) + EPS
                            lam = num / den if den > 0 else 1.0
                            Xmerge = lam * Xmerge

                    # Add to base (scale)
                    sl_base.add_(self.scale * Xmerge.to(sl_base.dtype))

                    cursor += take
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                merged += 1

        # Load merged state back
        if isinstance(base_model, nn.Module):
            model = base_model
            model.load_state_dict({k: v if not torch.is_floating_point(v) else v for k, v in base_cpu.items()}, strict=False)
        elif isinstance(base_model, LazyStateDict):
            model = base_model.meta_module.to_empty(device=base_model._device)
            result = model.load_state_dict({k: v for k, v in base_cpu.items()}, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected keys: {result.unexpected_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(base_model)}")

        self.print_profile_summary()
        if merged == 0:
            raise RuntimeError("No matching float tensors merged.")
        return model
