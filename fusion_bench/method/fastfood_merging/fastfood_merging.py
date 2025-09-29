# fastfood_merging.py
from __future__ import annotations
import math, hashlib
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


# ---------------- Fastfood / SRHT helpers ----------------
def _next_pow2(n: int) -> int:
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


# --------------- Zero-aware aggregation ----------------
@torch.no_grad()
def _ema_adaptive_beta(z_acc: Tensor, z_new: Tensor, gamma: float = 1.2, w_c: float = 0.6, w_s: float = 0.4, eps: float = 1e-8) -> float:
    """
    Compute adaptive β_t for EMA based on alignment and scale between accumulator and new task.
    
    Args:
        z_acc: Current accumulator vector
        z_new: New task vector to incorporate
        gamma: Sigmoid scaling factor
        w_c: Weight for cosine alignment term  
        w_s: Weight for scale ratio term
        eps: Small constant for numerical stability
    
    Returns:
        β_t ∈ [0,1]: mixing coefficient for EMA update
    """
    if z_acc.norm() < eps:
        return 0.0
    
    # Cosine alignment: how well z_new aligns with current accumulator
    c = (z_acc @ z_new) / (z_acc.norm() * z_new.norm() + eps)
    
    # Scale ratio: relative magnitude of accumulator vs new task
    s = z_acc.norm() / (z_new.norm() + eps)
    
    # Map to β via sigmoid (higher alignment & balanced scale → higher β)
    beta = torch.sigmoid(gamma * (w_c * c + w_s * s))
    return float(beta.item())


@torch.no_grad()
def _ema_merge_subspace(
    U: Tensor, 
    task_order: str = "given", 
    ema_gamma: float = 1.2,
    ema_w_c: float = 0.6, 
    ema_w_s: float = 0.4,
    weights: List[float] | None = None,
    custom_order: List[int] | None = None
) -> Tensor:
    """
    EMA merging in subspace with adaptive β_t.
    
    Args:
        U: [K, ..., M] stacked task vectors in subspace
        task_order: "given" | "random" | "cosine_similarity" | "custom"
        ema_gamma, ema_w_c, ema_w_s: EMA adaptive β parameters
        weights: Task importance weights (applied as scaling)
        custom_order: List of task indices when task_order="custom" (e.g., [2,0,1] to process tasks in that order)
    
    Returns:
        Merged vector in subspace
    """
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge")
    if K == 1:
        w = weights[0] if weights else 1.0
        return w * U[0]
    
    # Flatten for easier processing
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)  # [K, numel]
    
    # Determine processing order
    if task_order == "random":
        indices = torch.randperm(K).tolist()
    elif task_order == "custom":
        if custom_order is None:
            raise ValueError("custom_order must be provided when task_order='custom'")
        if len(custom_order) != K or set(custom_order) != set(range(K)):
            raise ValueError(f"custom_order must be a permutation of [0,1,...,{K-1}], got {custom_order}")
        indices = custom_order
    elif task_order == "cosine_similarity":
        # Start with first, then order remaining by cosine similarity to current accumulator
        indices = [0]
        z_acc = U_flat[0].clone()
        remaining = set(range(1, K))
        
        while remaining:
            # Find task most similar to current accumulator
            best_idx = None
            best_sim = -2.0  # cosine ∈ [-1,1]
            
            for idx in remaining:
                z_cand = U_flat[idx]
                if z_acc.norm() > 1e-8 and z_cand.norm() > 1e-8:
                    sim = (z_acc @ z_cand) / (z_acc.norm() * z_cand.norm())
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx
            
            if best_idx is not None:
                indices.append(best_idx)
                remaining.remove(best_idx)
                # Update accumulator for next iteration (simple average so far)
                z_acc = torch.stack([U_flat[i] for i in indices]).mean(dim=0)
            else:
                # Fallback: just take first remaining
                best_idx = next(iter(remaining))
                indices.append(best_idx)
                remaining.remove(best_idx)
    else:  # "given"
        indices = list(range(K))
    
    # EMA streaming update
    z_acc = torch.zeros_like(U_flat[0])
    
    for i, task_idx in enumerate(indices):
        z_new = U_flat[task_idx]
        
        # Apply task weight if provided
        if weights is not None:
            z_new = weights[task_idx] * z_new
        
        # Adaptive β based on current accumulator and new task
        beta = _ema_adaptive_beta(z_acc, z_new, ema_gamma, ema_w_c, ema_w_s)
        
        # EMA update: z_acc = β * z_acc + (1-β) * z_new
        z_acc = beta * z_acc + (1 - beta) * z_new
    
    return z_acc.view(orig_shape)


@torch.no_grad()
def _zero_aware_aggregate(
    U: Tensor, merge_func: str, weights: List[float] | None, **kwargs
) -> Tensor:
    """
    U: [K, ..., M] stacked donors in the chosen mixing space (subspace or postlift).
    merge_func ∈ {'sum','mean','max','signmax','ema'}:
      - 'sum'     : elementwise sum (weights optional)
      - 'mean'    : elementwise mean over *nonzero* contributors only (disentangled).
                    If weights given: (Σ w_k u_k 1[u_k!=0]) / (Σ w_k 1[u_k!=0])
      - 'max'     : elementwise argmax by |u_k|; if all zero → 0
      - 'signmax' : per position, pick the dominant sign across tasks (ignoring zeros),
                    then choose the largest |u_k| **with that sign**; on ties/no dominant
                    sign, fall back to 'max' behavior.
      - 'ema'     : Exponential Moving Average with adaptive β_t based on alignment & scale.
                    Processes tasks sequentially with order-dependent results.
                    Task ordering: "given" (modelpool order), "random" (shuffle), 
                    "cosine_similarity" (similar tasks first), "custom" (user-specified order)
    """
    K = U.shape[0]
    mf = merge_func.lower()
    if mf not in {"sum", "mean", "max", "signmax", "ema"}:
        raise ValueError(f"merge_func={merge_func} not in {{'sum','mean','max','signmax','ema'}}")

    if mf == "ema":
        return _ema_merge_subspace(
            U, 
            task_order=kwargs.get("ema_task_order", "given"),
            ema_gamma=kwargs.get("ema_gamma", 1.2),
            ema_w_c=kwargs.get("ema_w_c", 0.6),
            ema_w_s=kwargs.get("ema_w_s", 0.4),
            weights=weights,
            custom_order=kwargs.get("ema_custom_order", None)
        )

    if mf == "sum":
        if weights is None:
            return U.sum(dim=0)
        w = torch.tensor(weights, dtype=U.dtype, device=U.device).view(
            K, *([1] * (U.ndim - 1))
        )
        return (w * U).sum(dim=0)

    mask = (U != 0)
    if mf == "mean":
        if weights is None:
            denom = mask.sum(dim=0).clamp_min(1)
            return (U * mask).sum(dim=0) / denom
        w = torch.tensor(weights, dtype=U.dtype, device=U.device).view(
            K, *([1] * (U.ndim - 1))
        )
        num = (w * U * mask).sum(dim=0)
        den = (w * mask).sum(dim=0).clamp_min(1e-12)
        return num / den

    absU = U.abs()

    if mf == "max":
        idx = absU.argmax(dim=0)
        out = torch.gather(U, dim=0, index=idx.unsqueeze(0)).squeeze(0)
        all_zero = (absU.sum(dim=0) == 0)
        if all_zero.any():
            out = out.masked_fill(all_zero, 0.0)
        return out

    if mf == "signmax":
        sgn = torch.sign(U)                 # {-1, 0, +1}
        pos_count = (sgn > 0).sum(dim=0)
        neg_count = (sgn < 0).sum(dim=0)

        dom_pos = pos_count > neg_count     # positive majority
        dom_neg = neg_count > pos_count     # negative majority
        tie_or_none = ~(dom_pos | dom_neg)  # tie or all zeros

        match_pos = (U > 0) & dom_pos.unsqueeze(0)
        match_neg = (U < 0) & dom_neg.unsqueeze(0)
        match = match_pos | match_neg

        masked_mag = absU.clone()
        masked_mag[~match] = -float("inf")

        idx_dom = masked_mag.argmax(dim=0)
        idx_fallback = absU.argmax(dim=0)
        idx = torch.where(tie_or_none, idx_fallback, idx_dom)

        out = torch.gather(U, dim=0, index=idx.unsqueeze(0)).squeeze(0)

        all_zero = (absU.sum(dim=0) == 0)
        if all_zero.any():
            out = out.masked_fill(all_zero, 0.0)
        return out

def _layer_key(name: str) -> str:
    """Heuristic layer-grouping key (works for most HF models)."""
    parts = name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


# ------------------ Algorithm ------------------
@auto_register_config
class FastfoodSubspaceMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Task-vector merging via Fastfood/SRHT subspaces.

    Controls:
      subspace_scope: "per_tensor" | "layer" | "global"
      merge_where:    "subspace" | "postlift"
      merge_func:     "sum" | "mean" | "max" | "signmax" | "ema"  (zero-aware & disentangled)
      proj_ratio:     float (0..1)
      use_G:          bool
      block_rows:     int
      weights:        list[float] (donor weights; normalized internally)
      scale:          float (post-merge scale on Δ*)
      
      # EMA-specific parameters (when merge_func="ema"):
      ema_task_order: "given" | "random" | "cosine_similarity" | "custom"
      ema_gamma:      float (sigmoid scaling factor, default 1.2)
      ema_w_c:        float (cosine alignment weight, default 0.6)
      ema_w_s:        float (scale ratio weight, default 0.4)
      ema_custom_order: list[str] (task names in desired order, when ema_task_order="custom")
      
      ties_trim_pct, tadrop_tau, use_pareto: kept for API compatibility (unused)
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",  # "per_tensor" | "layer" | "global"
        merge_where: str = "subspace",   # "subspace" | "postlift"
        merge_func: str = "sum",         # "sum" | "mean" | "max" | "signmax" | "ema"
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        # EMA-specific parameters
        ema_task_order: str = "given",   # "given" | "random" | "cosine_similarity" | "custom"
        ema_gamma: float = 1.2,          # sigmoid scaling factor
        ema_w_c: float = 0.6,            # cosine alignment weight
        ema_w_s: float = 0.4,            # scale ratio weight
        ema_custom_order: List[str] | None = None,  # task names in desired order (when ema_task_order="custom")
        # Kept in signature for compatibility (not used since align logic removed)
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.0,
        use_pareto: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        assert merge_where in {"subspace", "postlift"}
        self.merge_where = merge_where
        self.merge_func = str(merge_func).lower()
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        
        # EMA parameters
        self.ema_task_order = str(ema_task_order)
        self.ema_gamma = float(ema_gamma)
        self.ema_w_c = float(ema_w_c)
        self.ema_w_s = float(ema_w_s)
        self.ema_custom_order = list(ema_custom_order) if ema_custom_order is not None else None

    # ------------------- main -------------------
    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        modelpool = to_modelpool(modelpool)

        # ---------- Load ----------
        with self.profile("loading models"):
            base_model = modelpool.load_model("_pretrained_")
            donor_names = list(modelpool.model_names)
            if len(donor_names) < 2:
                raise ValueError(f"Need ≥2 donors; got {len(donor_names)}")

            donors_sd: List[StateDictType] = [
                modelpool.load_model(n).state_dict(keep_vars=True)
                for n in donor_names
            ]
            base_sd: Dict[str, Tensor] = base_model.state_dict(keep_vars=True)

        # ---------- Eligible tensors ----------
        keys_all = list(base_sd.keys())
        keys_float = [
            k for k in keys_all
            if (k in donors_sd[0])
            and torch.is_floating_point(base_sd[k])
            and base_sd[k].ndim >= 1
            and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
        ]
        K = len(donor_names)
        print(f"[Setup] donors={K} | total tensors={len(keys_all)} | eligible float tensors={len(keys_float)}")

        if not keys_float:
            raise RuntimeError("No overlapping float tensors with identical shapes. Nothing to merge.")

        # ---------- Weights ----------
        if self.weights is None:
            w = [1.0 / K] * K
        else:
            if len(self.weights) != K:
                raise ValueError("`weights` length must match number of donors.")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]

        # ---------- EMA Custom Order Mapping ----------
        ema_custom_indices = None
        if self.merge_func == "ema" and self.ema_task_order == "custom":
            if self.ema_custom_order is None:
                raise ValueError("ema_custom_order must be provided when ema_task_order='custom'")
            
            # Map task names to indices
            name_to_idx = {name: i for i, name in enumerate(donor_names)}
            try:
                ema_custom_indices = [name_to_idx[task_name] for task_name in self.ema_custom_order]
            except KeyError as e:
                available_names = list(donor_names)
                raise ValueError(f"Task name {e} in ema_custom_order not found in donor names. Available: {available_names}")
            
            if len(ema_custom_indices) != K:
                raise ValueError(f"ema_custom_order must include all {K} tasks, got {len(ema_custom_indices)}")
            if set(ema_custom_indices) != set(range(K)):
                raise ValueError(f"ema_custom_order must be a permutation of all task indices, got {ema_custom_indices}")
            
            print(f"[EMA] Custom task order: {self.ema_custom_order} -> indices {ema_custom_indices}")

        # ---------- Seed scoping ----------
        def proj_seed_key(param_name: str) -> str:
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            if self.subspace_scope == "layer":
                return _layer_key(param_name)
            return param_name  # per_tensor

        # Determine global D (max last-dim) if needed
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for k in keys_float:
                t = base_sd[k]
                maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd

        # Report subspace sizing (first few examples)
        def _dim_for(k: str) -> Tuple[int, int]:
            d_last = int(base_sd[k].shape[-1])
            cur_D = global_D if (global_D is not None) else d_last
            m = max(1, int(cur_D * self.proj_ratio))
            return cur_D, m

        ex = keys_float[:5]
        if ex:
            dims = [(_dim_for(k), k) for k in ex]
            print("[Dims] scope={} | proj_ratio={:.3f} | examples:".format(self.subspace_scope, self.proj_ratio))
            for (D, m), k in dims:
                print(f"   - {k}: original_last_dim={int(base_sd[k].shape[-1])} | scoped_dim={D} → proj_dim={m} (compression={m/max(1,D):.3f})")
        
        # Show EMA parameters if using EMA
        if self.merge_func == "ema":
            print(f"[EMA] task_order={self.ema_task_order} | gamma={self.ema_gamma:.3f} | w_c={self.ema_w_c:.3f} | w_s={self.ema_w_s:.3f}")

        # ---------- Work on CPU copies ----------
        base_cpu = {k: v.detach().cpu().clone() for k, v in base_sd.items()}
        donors_cpu = [{k: v.detach().cpu().clone() for k, v in d.items()} for d in donors_sd]
        dev = self.device

        # ---------- Merge ----------
        merged_tensors = 0
        changed_params = 0

        # operator cache keyed by (seed_key, cur_D, proj_dim)
        op_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}

        # Small sample for lift error (subspace only)
        lift_err_num = 0.0
        lift_err_den = 0.0

        with self.profile("merging models"):
            for name in keys_float:
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # choose scoped dim & build (or reuse) operator
                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                proj_dim = max(1, int(cur_D * self.proj_ratio))
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = _fastfood_ops(
                        cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # donor deltas aligned to cur_D if global scope
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor:cursor + take, :]
                        delta = sl_donor - sl_base
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros((take, cur_D), dtype=torch.float32, device="cpu")
                            buf[:, :d_last].copy_(delta)
                            Xs.append(buf)
                        else:
                            Xs.append(delta)

                    # project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # mix in chosen space
                    if self.merge_where == "postlift":
                        # reconstruct donors first, then aggregate in original space
                        Xhats = [lift(Y).to("cpu", non_blocking=True)[:, :d_last] for Y in Ys]
                        U_stack = torch.stack(Xhats, dim=0)  # [K, take, d]
                        Xmerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        ).to("cpu", non_blocking=True)
                    else:
                        # aggregate in subspace, then lift once
                        U_stack = torch.stack(Ys, dim=0)  # [K, take, m]
                        Ymerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        )
                        Xmerge_full = lift(Ymerge).to("cpu", non_blocking=True)  # [take, cur_D]
                        Xmerge = Xmerge_full[:, :d_last]

                        # accumulate lift reconstruction error stats on a tiny slice
                        if take > 0 and (lift_err_den < 1e8):  # guard cost
                            # pick the first donor to estimate lift error
                            X0 = Xs[0].to(dev, non_blocking=True)
                            Y0 = Ys[0]
                            X0_rec = lift(Y0).to("cpu", non_blocking=True)[:, :d_last]
                            diff = (X0_rec.to(torch.float32) - Xs[0][:, :d_last].to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(Xs[0][:, :d_last].pow(2).sum().item())

                    # write back (scale)
                    upd = (self.scale * Xmerge).to(sl_base.dtype)
                    sl_base.add_(upd)

                    # did anything change?
                    tensor_changed = tensor_changed or bool(upd.abs().max().item() > 0)

                    cursor += take
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                merged_tensors += 1
                if tensor_changed:
                    changed_params += 1

        # ---------- Stats / sanity ----------
        # scan NaN/Inf
        bad, total_float = [], 0
        for n, p in base_cpu.items():
            if not torch.is_floating_point(p):
                continue
            total_float += 1
            if torch.isnan(p).any() or torch.isinf(p).any():
                bad.append(n)

        # scale drift vs original base
        drift_hi, drift_lo = [], []
        for k in keys_float:
            pre = base_sd[k].float()
            now = base_cpu[k].float()
            nb = float(pre.norm().item()) + 1e-12
            na = float(now.norm().item())
            ratio = na / nb
            if ratio > 10.0:
                drift_hi.append((k, ratio))
            elif ratio < 0.1:
                drift_lo.append((k, ratio))

        drift_hi.sort(key=lambda x: x[1], reverse=True)
        drift_lo.sort(key=lambda x: x[1])

        # print summary
        print("\n=== Merge Summary ===")
        print(f"[Summary] donors={K} | eligible_tensors={len(keys_float)} | processed={merged_tensors} | changed_tensors={changed_params}")
        if bad:
            print(f"[Summary] ⚠️ NaN/Inf in {len(bad)}/{total_float} float tensors (showing up to 10):")
            for n in bad[:10]:
                print("  -", n)
        else:
            print(f"[Summary] ✓ No NaN/Inf across {total_float} float tensors.")

        if self.merge_where == "subspace" and lift_err_den > 0:
            rel = math.sqrt(lift_err_num) / (math.sqrt(lift_err_den) + EPS)
            print(f"[Summary] Lift reconstruction rel. error (Fro): {rel:.6f}")
        else:
            print("[Summary] Lift reconstruction error: N/A (postlift mixing or no samples).")

        print(f"[Summary] Large ↑ drift (>10x): {len(drift_hi)} | Large ↓ drift (<0.1x): {len(drift_lo)}")
        for name, r in drift_hi[:8]:
            print(f"   HI  {r:.3f}  {name}")
        for name, r in drift_lo[:8]:
            print(f"   LO  {r:.3f}  {name}")

        self.print_profile_summary()

        # ---------- Load merged state back ----------
        if isinstance(base_model, nn.Module):
            model = base_model
            model.load_state_dict(
                {k: v if not torch.is_floating_point(v) else v for k, v in base_cpu.items()},
                strict=False,
            )
        elif isinstance(base_model, LazyStateDict):
            model = base_model.meta_module.to_empty(device=base_model._device)
            result = model.load_state_dict({k: v for k, v in base_cpu.items()}, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected keys: {result.unexpected_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(base_model)}")

        if merged_tensors == 0:
            raise RuntimeError("No tensors were processed; check eligibility filters.")
        if changed_params == 0:
            print("⚠️ Note: processed tensors but no numeric changes detected (donor deltas may be zero).")

        return model
