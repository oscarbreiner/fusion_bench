from __future__ import annotations
import math
import hashlib
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


def _seed_from(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:4], "little")


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
    The same (B, G, Π, P) are reused for all donors sharing `seed_key`.
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

    # JL row subset and scaling
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


# --------------- Alignment / masks / α solving ---------------
@torch.no_grad()
def _ties_sign_and_trim(U: Tensor, trim_pct: float = 0.0) -> Tensor:
    """
    U: [K, N, M] (K donors stacked) or [K, M]
    Elect global sign per-coordinate by column-sum and optionally trim by magnitude.
    """
    # Collapse donors to elected sign using sum over donor axis K
    sign = torch.sign(U.sum(dim=0))
    sign[sign == 0] = 1.0
    U = U * sign  # broadcast over donors

    if trim_pct and trim_pct > 0.0:
        q = max(0.0, min(100.0, float(trim_pct)))
        thr = torch.quantile(U.abs().reshape(-1), q / 100.0)
        U = U * (U.abs() >= thr)
    return U


@torch.no_grad()
def _tadrop_mask(U: Tensor, tau: float = 0.5, eps: float = EPS) -> Tensor:
    """
    Tensor-wise Adaptive Drop around mean reference.
    U: [K, N, M] or [K, M]
    """
    ref = U.mean(dim=0, keepdim=True)
    denom = U.abs() + ref.abs() + eps
    d = (U - ref).abs() / denom
    m = (d <= tau).to(U.dtype)
    return U * m


@torch.no_grad()
def _pareto_weights(Ys: List[Tensor]) -> List[float]:
    """
    Two-donor min-norm convex combination in the mixing space.
    """
    if len(Ys) != 2:
        return [1.0 / len(Ys)] * len(Ys)
    U, V = Ys[0], Ys[1]
    D = (V - U)
    num = (D * V).sum()
    den = (D * D).sum() + EPS
    a = torch.clamp(num / den, 0.0, 1.0).item()
    return [1.0 - a, a]


# --------------- Zero-aware aggregation ----------------
@torch.no_grad()
def _zero_aware_aggregate(
    U: Tensor, merge_func: str, weights: List[float] | None
) -> Tensor:
    """
    U: [K, ..., M] stacked donors in the chosen mixing space (subspace or postlift).
    merge_func ∈ {'sum','mean','max'}:
      - 'sum'  : elementwise sum (zeros naturally neutral)
      - 'mean' : elementwise mean over *nonzero* contributors only (disentangled).
                 If weights given: (Σ w_k u_k 1[u_k!=0]) / (Σ w_k 1[u_k!=0])
      - 'max'  : elementwise argmax by |u_k|; if all zero → 0
    """
    K = U.shape[0]
    mf = merge_func.lower()
    if mf not in {"sum", "mean", "max"}:
        raise ValueError(f"merge_func={merge_func} not in {{'sum','mean','max'}}")

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

    # mf == 'max'
    absU = U.abs()
    idx = absU.argmax(dim=0)
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
    General subspace merge with Fastfood/SRHT operators.

    Options:
      subspace_scope: "per_tensor" | "layer" | "global"
      merge_where:    "subspace" | "postlift"
      align_mode:     "none" | "ties" | "tadrop"
      ties_trim_pct:  float (0..100)
      tadrop_tau:     float
      merge_func:     "sum" | "mean" | "max"  (zero-aware & disentangled)
      use_pareto:     bool (for 2 donors)
      use_rescale:    bool (EMR-R style magnitude match)
      proj_ratio:     float (0..1)
      use_G:          bool
      block_rows:     int
      weights:        list[float] (donor weights; normalized internally)
      scale:          float (post-merge scale on Δ*)
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",  # per_tensor | layer | global
        merge_where: str = "subspace",  # subspace | postlift
        align_mode: str = "none",  # none | ties | tadrop
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.5,
        merge_func: str = "sum",  # sum | mean | max
        use_pareto: bool = False,
        use_rescale: bool = False,
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        self.merge_where = str(merge_where)
        self.align_mode = str(align_mode)
        self.ties_trim_pct = float(ties_trim_pct)
        self.tadrop_tau = float(tadrop_tau)
        self.merge_func = str(merge_func).lower()
        self.use_pareto = bool(use_pareto)
        self.use_rescale = bool(use_rescale)
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        modelpool = to_modelpool(modelpool)

        # ---------- Load models ----------
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

        # ---------- Weights ----------
        if self.weights is None:
            w = [1.0 / len(donor_names)] * len(donor_names)
        else:
            if len(self.weights) != len(donor_names):
                raise ValueError("`weights` length must match number of donors.")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]

        # ---------- Seed scoping ----------
        def proj_seed_key(param_name: str) -> str:
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            if self.subspace_scope == "layer":
                return _layer_key(param_name)
            return param_name  # per_tensor

        # Determine global D (max last-dim)
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for _, t in base_sd.items():
                if torch.is_floating_point(t) and t.ndim >= 1:
                    maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd

        # Work on CPU copies for safety
        base_cpu = {k: v.detach().cpu() for k, v in base_sd.items()}
        donors_cpu = [{k: v.detach().cpu() for k, v in d.items()} for d in donors_sd]
        dev = self.device
        merged_tensors = 0

        # ---------- Merge ----------
        with self.profile("merging models"):
            # cache operators keyed by (seed_key, cur_D, proj_dim)
            op_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}

            for name, tb in base_cpu.items():
                if not torch.is_floating_point(tb) or tb.ndim < 1:
                    continue
                shape = tb.shape
                if not all(
                    (name in d)
                    and torch.is_floating_point(d[name])
                    and d[name].shape == shape
                    for d in donors_cpu
                ):
                    continue

                d_last = int(shape[-1])
                rows = tb.numel() // d_last
                if rows == 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                proj_dim = max(1, int(cur_D * self.proj_ratio))
                cache_key = (seed_key, cur_D, proj_dim)

                if cache_key not in op_cache:
                    fwd, lift = _fastfood_ops(
                        cur_D,
                        proj_dim,
                        seed_key=seed_key,
                        device=dev,
                        use_G=self.use_G,
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                cursor = 0
                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor : cursor + take, :]

                    # donor deltas aligned to cur_D if global scope
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        sl_donor = dsd[name].view(rows, d_last).float()[
                            cursor : cursor + take, :
                        ]
                        delta = sl_donor - sl_base
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros(
                                (take, cur_D), dtype=torch.float32, device="cpu"
                            )
                            buf[:, :d_last].copy_(delta)
                            Xs.append(buf)
                        else:
                            Xs.append(delta)

                    # project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # choose mixing space
                    if self.merge_where == "postlift":
                        Xhats = [
                            lift(Y).to("cpu", non_blocking=True)[:, :d_last] for Y in Ys
                        ]
                        mix_space = "postlift"
                    else:
                        mix_space = "subspace"

                    # alignment / masking
                    if self.align_mode != "none":
                        if mix_space == "subspace":
                            U = torch.stack(Ys, dim=0)  # [K, take, m]
                        else:
                            U = torch.stack(Xhats, dim=0)  # [K, take, d]

                        if self.align_mode == "ties":
                            U = _ties_sign_and_trim(U, trim_pct=self.ties_trim_pct)
                        elif self.align_mode == "tadrop":
                            U = _tadrop_mask(U, tau=self.tadrop_tau)

                        if mix_space == "subspace":
                            Ys = [U[i] for i in range(U.shape[0])]
                        else:
                            Xhats = [U[i] for i in range(U.shape[0])]

                    # weights (optional Pareto for 2 donors)
                    if self.use_pareto and len(Ys) == 2:
                        cur_w = (
                            _pareto_weights(Ys)
                            if mix_space == "subspace"
                            else _pareto_weights(Xhats)
                        )
                    else:
                        cur_w = w

                    # aggregate in chosen space (zero-aware)
                    if mix_space == "subspace":
                        U_stack = torch.stack(Ys, dim=0)  # [K, take, m]
                        Ymerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=cur_w if self.merge_func in {"sum", "mean"} else None,
                        )
                        Xmerge = lift(Ymerge).to("cpu", non_blocking=True)[:, :d_last]
                        if self.use_rescale:
                            num = 0.0
                            for wi, Xorig in zip(cur_w, Xs):
                                num += wi * float(torch.linalg.vector_norm(Xorig))
                            den = float(torch.linalg.vector_norm(Xmerge)) + EPS
                            lam = num / den if den > 0 else 1.0
                            Xmerge = lam * Xmerge
                    else:
                        U_stack = torch.stack(Xhats, dim=0)  # [K, take, d]
                        Xmerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=cur_w if self.merge_func in {"sum", "mean"} else None,
                        ).to("cpu", non_blocking=True)
                        if self.use_rescale:
                            num = 0.0
                            for wi, Xi in zip(cur_w, Xhats):
                                num += wi * float(torch.linalg.vector_norm(Xi))
                            den = float(torch.linalg.vector_norm(Xmerge)) + EPS
                            lam = num / den if den > 0 else 1.0
                            Xmerge = lam * Xmerge

                    # write back (scale)
                    sl_base.add_(self.scale * Xmerge.to(sl_base.dtype))

                    cursor += take
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                merged_tensors += 1

        # ---------- Stats / sanity ----------
        try:
            drift_hi, drift_lo = [], []
            total_params, changed = 0, 0
            for k, vb in base_sd.items():
                if not torch.is_floating_point(vb) or vb.ndim < 1:
                    continue
                v_after = base_cpu[k].float()
                nb = float(vb.float().norm().item()) + 1e-12
                na = float(v_after.norm().item())
                ratio = na / nb
                total_params += 1
                if ratio > 10.0:
                    drift_hi.append((k, ratio))
                elif ratio < 0.1:
                    drift_lo.append((k, ratio))
                if (v_after - vb.float()).abs().sum().item() > 0:
                    changed += 1

            drift_hi.sort(key=lambda x: x[1], reverse=True)
            drift_lo.sort(key=lambda x: x[1])

            print(f"\n[Stats] Tensors merged: {changed}/{total_params}")
            print(
                f"[Stats] Large ↑ drift (>10x): {len(drift_hi)} | Large ↓ drift (<0.1x): {len(drift_lo)}"
            )
            for name, r in drift_hi[:10]:
                print(f"   HI  {r:.3f}  {name}")
            for name, r in drift_lo[:10]:
                print(f"   LO  {r:.3f}  {name}")

            self.print_profile_summary()
        except Exception as e:
            print("[Stats] Skipped due to:", repr(e))

        # ---------- Load merged state back ----------
        if isinstance(base_model, nn.Module):
            model = base_model
            # replace tensors from base_cpu back into model
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
            raise RuntimeError("No matching float tensors merged.")
        return model
