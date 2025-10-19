"""
Adaptive Subspace Budget (Per-Tensor)

This analysis estimates, for EACH parameter tensor independently, the
effective rank along the *last dimension* (the axis you project in Fastfood/SRHT),
using all donor task deltas stacked across tasks and rows.

It outputs a suggested projection size m_t per tensor:
    m_t = clamp( ceil(beta * r_eff), m_min, floor(m_max_frac * d_last) )

Key properties:
- No concatenation across tensors or layers.
- Mirrors your per-tensor, row-wise projection used in merging.
- Works for Linear (out×in), Conv (out×in·k·k), and generic 2D weights.
- Efficient: accumulates C = X^T X in blocks (no giant X materialization).
"""

from __future__ import annotations
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


# ---------------------------
# Small helpers
# ---------------------------

def _extract_layer_number(param_name: str) -> int:
    # Reuse the robust patterns from your existing analysis
    for pat in (r'\.resblocks\.(\d+)\.', r'\.layers\.(\d+)\.', r'\.layer\.(\d+)\.', r'\.blocks?\.(\d+)\.'):
        m = re.search(pat, param_name)
        if m:
            return int(m.group(1))
    return -1


def _should_use_tensor(name: str, tensor: torch.Tensor,
                       include_1d: bool, include_conv: bool,
                       include_linear: bool, include_other_2d: bool) -> bool:
    if tensor.ndim == 1:
        return bool(include_1d)
    if tensor.ndim == 2:
        # Try to infer linear-like
        if include_linear:
            return True
        return bool(include_other_2d)
    if tensor.ndim == 4:
        # Conv kernel
        return bool(include_conv)
    # Skip exotic shapes by default
    return False


def _to_rows_lastdim(t: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    """
    Map parameter to [rows, d_last] without copies when possible.
      - 2D: (out, in) -> rows=out, d_last=in
      - 4D: (out, in, k, k) -> rows=out, d_last=in*k*k
      - 1D: rows=1, d_last=len (optional)
    Returns (rows, d_last, view)
    """
    if t.ndim == 2:
        out, inn = t.shape
        return int(out), int(inn), t.view(out, inn)
    if t.ndim == 4:
        out, c, k1, k2 = t.shape
        d_last = int(c * k1 * k2)
        return int(out), d_last, t.reshape(out, d_last)
    if t.ndim == 1:
        return 1, int(t.numel()), t.view(1, -1)
    raise ValueError(f"Unsupported ndim={t.ndim} for tensor of shape {tuple(t.shape)}")


@torch.no_grad()
def _effective_rank_from_cov_eigs(evals: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank on variance spectrum λ (eigenvalues of X^T X).
    r_eff = exp( -Σ p_i log p_i ), p_i = λ_i / Σ λ_i
    """
    evals = evals.clamp_min(0.0).to(torch.float64)
    s = evals.sum()
    if s <= 0:
        return 0.0
    p = (evals / s).clamp_min(eps)
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H))


@torch.no_grad()
def _accumulate_cov_lastdim(deltas_per_task: List[torch.Tensor],
                            max_rows_per_task: Optional[int],
                            row_block: int,
                            use_dtype: torch.dtype) -> torch.Tensor:
    """
    Accumulate C = X^T X along last dim for a single parameter across all tasks.
    X stacks rows from all tasks. Processes in blocks to save memory.
    Returns C as float64 for stable eigendecomposition.
    """
    C: Optional[torch.Tensor] = None
    for Δ in deltas_per_task:
        Δ = Δ.to(dtype=use_dtype)
        rows, d_last, view = _to_rows_lastdim(Δ)
        # optional row subsampling per task
        if (max_rows_per_task is not None) and (rows > max_rows_per_task):
            # simple uniform subsample without replacement
            idx = torch.randperm(rows)[:max_rows_per_task]
            view_iter = view[idx]
            rows_iter = int(max_rows_per_task)
        else:
            view_iter = view
            rows_iter = rows

        # block over rows
        for i in range(0, rows_iter, row_block):
            sl = view_iter[i:i + row_block, :]              # [b, d_last]
            Ct = sl.t().matmul(sl)                          # [d_last, d_last]
            C = Ct if C is None else (C + Ct)
    if C is None:
        raise RuntimeError("No data to accumulate covariance.")
    return C.to(torch.float64)


def _suggest_m_from_rank(r_eff: float, d_last: int,
                         beta: float, m_min: int, m_max_frac: float) -> int:
    m_max = max(1, int(math.floor(m_max_frac * d_last)))
    m = int(math.ceil(beta * max(0.0, r_eff)))
    return max(1, min(max(m_min, m), m_max, d_last))


# ---------------------------
# Main analysis
# ---------------------------

@auto_register_config
class AdaptiveSubspaceBudgetAnalysis(LightningFabricMixin, BaseAlgorithm):
    """
    Per-tensor effective rank (last-dim) -> suggested projection size m_t.

    For each tensor W (ndim>=2) and across all donor tasks:
      1) Build ΔW = W_finetuned - W_base
      2) Accumulate C = X^T X where X stacks rows of ΔW across tasks (and rows),
         matching the projection axis (last dimension).
      3) Eig(C) -> λ; effective rank r_eff from λ (variance spectrum).
      4) Suggest m_t = clamp( ceil(beta * r_eff), m_min, floor(m_max_frac * d_last) ).

    Outputs CSV + JSON/PT maps and diagnostic plots.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",

        # mapping r_eff -> m
        beta: float = 2.0,
        m_min: int = 16,
        m_max_frac: float = 0.5,

        # efficiency
        row_block: int = 8192,
        max_rows_per_task: Optional[int] = None,
        dtype: str = "float32",

        # inclusion flags
        include_conv: bool = True,
        include_linear: bool = True,
        include_other_2d: bool = True,
        include_1d: bool = False,

        # IO
        save_csv: bool = True,
        save_json: bool = True,
        save_pt: bool = True,
        create_plots: bool = True,

        # misc
        warn_on_empty: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.trainable_only = bool(trainable_only)
        self._output_path = output_path
        self.method_name = method_name or "adaptive_subspace_budget"
        self.device = torch.device(device)

        self.beta = float(beta)
        self.m_min = int(m_min)
        self.m_max_frac = float(m_max_frac)

        self.row_block = int(row_block)
        self.max_rows_per_task = max_rows_per_task if (max_rows_per_task is None) else int(max_rows_per_task)
        self.in_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)

        self.include_conv = bool(include_conv)
        self.include_linear = bool(include_linear)
        self.include_other_2d = bool(include_other_2d)
        self.include_1d = bool(include_1d)

        self.save_csv = bool(save_csv)
        self.save_json = bool(save_json)
        self.save_pt = bool(save_pt)
        self.create_plots = bool(create_plots)

        self.warn_on_empty = bool(warn_on_empty)

    @property
    def output_path(self) -> str:
        if self._output_path is None:
            return self.fabric.logger.log_dir
        return self._output_path

    # ---------------------------
    # State helpers
    # ---------------------------
    def _state(self, model: nn.Module) -> StateDictType:
        return trainable_state_dict(model) if self.trainable_only else model.state_dict()

    # ---------------------------
    # Orchestrator
    # ---------------------------
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        log.info("Starting Adaptive Subspace Budget Analysis (per-tensor)")
        os.makedirs(self.output_path, exist_ok=True)

        # Load base
        base = modelpool.load_pretrained_model()
        base_sd = self._state(base)

        # Collect donors + their deltas
        donors: Dict[str, nn.Module] = dict(modelpool.named_models())
        donor_names = list(donors.keys())
        if len(donor_names) == 0:
            raise RuntimeError("No donor models found in modelpool.")

        donors_sd = {name: self._state(donors[name]) for name in donor_names}

        # Build per-tensor list of deltas across tasks
        # Keep only tensors that appear in base and all donors and match shape
        common_names: List[str] = []
        for k, tb in base_sd.items():
            if not torch.is_floating_point(tb):
                continue
            if not all((k in donors_sd[n]) and (donors_sd[n][k].shape == tb.shape) and torch.is_floating_point(donors_sd[n][k]) for n in donor_names):
                continue
            if not _should_use_tensor(k, tb, self.include_1d, self.include_conv, self.include_linear, self.include_other_2d):
                continue
            common_names.append(k)

        if not common_names and self.warn_on_empty:
            log.warning("No eligible tensors found for analysis (check inclusion flags).")

        results_rows: List[Dict[str, Any]] = []
        suggest_map: Dict[str, int] = {}

        # Main loop
        for name in tqdm(common_names, desc="Per-tensor effective rank (last-dim)"):
            t_base = base_sd[name].detach().cpu()
            layer_idx = _extract_layer_number(name)

            # Gather task deltas for this tensor
            deltas: List[torch.Tensor] = []
            for dn in donor_names:
                t_donor = donors_sd[dn][name].detach().cpu()
                deltas.append((t_donor - t_base).to(self.in_dtype))

            # shape info
            rows, d_last, _ = _to_rows_lastdim(t_base)
            if d_last <= 0:
                continue

            # accumulate covariance C = X^T X in float64
            try:
                C = _accumulate_cov_lastdim(
                    deltas_per_task=deltas,
                    max_rows_per_task=self.max_rows_per_task,
                    row_block=self.row_block,
                    use_dtype=self.in_dtype,
                )
            except RuntimeError as e:
                log.warning(f"[skip] {name}: {e}")
                continue

            # eigenvalues and r_eff
            evals = torch.linalg.eigvalsh(C)  # float64
            r_eff = _effective_rank_from_cov_eigs(evals)

            # map to m_t
            m_t = _suggest_m_from_rank(
                r_eff=r_eff,
                d_last=d_last,
                beta=self.beta,
                m_min=self.m_min,
                m_max_frac=self.m_max_frac,
            )

            # record
            results_rows.append({
                "param_name": name,
                "layer_index": layer_idx,
                "rows": int(rows),
                "d_last": int(d_last),
                "r_eff_lastdim": float(r_eff),
                "beta": float(self.beta),
                "m_min": int(self.m_min),
                "m_max_frac": float(self.m_max_frac),
                "m_suggested": int(m_t),
            })
            suggest_map[name] = int(m_t)

        # Save tabular results
        if self.save_csv and results_rows:
            df = pd.DataFrame(results_rows)
            csv_path = os.path.join(self.output_path, f"adaptive_subspace_budget_{self.method_name}.csv")
            df.to_csv(csv_path, index=False)
            log.info(f"Saved per-tensor budget CSV to {csv_path}")

        # Save mapping JSON / PT
        if self.save_json and suggest_map:
            json_path = os.path.join(self.output_path, f"adaptive_subspace_budget_{self.method_name}.json")
            with open(json_path, "w") as f:
                json.dump(suggest_map, f, indent=2)
            log.info(f"Saved suggested m map (JSON) to {json_path}")

        if self.save_pt and suggest_map:
            pt_path = os.path.join(self.output_path, f"adaptive_subspace_budget_{self.method_name}.pt")
            torch.save(suggest_map, pt_path)
            log.info(f"Saved suggested m map (PT) to {pt_path}")

        # Optional plots
        if self.create_plots and results_rows:
            self._make_plots(pd.DataFrame(results_rows))

        log.info("Adaptive Subspace Budget Analysis complete.")
        # return base (to match BaseAlgorithm API)
        return base

    # ---------------------------
    # Plots
    # ---------------------------
    def _make_plots(self, df: pd.DataFrame):
        try:
            os.makedirs(self.output_path, exist_ok=True)
            # Histogram of r_eff
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df["r_eff_lastdim"].values, bins=50, alpha=0.85)
            ax.set_title("Per-tensor Effective Rank (last-dim)")
            ax.set_xlabel("r_eff")
            ax.set_ylabel("#tensors")
            p1 = os.path.join(self.output_path, f"hist_reff_{self.method_name}.pdf")
            fig.tight_layout()
            fig.savefig(p1, dpi=300)
            plt.close(fig)

            # Scatter: d_last vs r_eff and suggested m
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
            ax2[0].scatter(df["d_last"].values, df["r_eff_lastdim"].values, s=8, alpha=0.6)
            ax2[0].set_xlabel("d_last")
            ax2[0].set_ylabel("r_eff (last-dim)")
            ax2[0].set_title("d_last vs r_eff")

            ax2[1].scatter(df["d_last"].values, df["m_suggested"].values, s=8, alpha=0.6)
            ax2[1].set_xlabel("d_last")
            ax2[1].set_ylabel("m_suggested")
            ax2[1].set_title("d_last vs m_suggested")

            p2 = os.path.join(self.output_path, f"scatter_dims_{self.method_name}.pdf")
            fig2.tight_layout()
            fig2.savefig(p2, dpi=300)
            plt.close(fig2)

            # Boxplot by layer (optional; robust summary)
            if "layer_index" in df.columns:
                # drop unknown layers (-1)
                dff = df[df["layer_index"] >= 0]
                if len(dff) > 0:
                    # aggregate per layer for visibility
                    groups = dff.groupby("layer_index")["r_eff_lastdim"]
                    med = groups.median()
                    q1 = groups.quantile(0.25)
                    q3 = groups.quantile(0.75)
                    fig3, ax3 = plt.subplots(figsize=(12, 4))
                    ax3.plot(med.index, med.values, label="median r_eff", marker="o")
                    ax3.fill_between(med.index, q1.values, q3.values, alpha=0.2, label="IQR")
                    ax3.set_xlabel("Layer")
                    ax3.set_ylabel("r_eff (last-dim)")
                    ax3.set_title("Per-tensor r_eff by Layer (median ± IQR)")
                    ax3.legend()
                    p3 = os.path.join(self.output_path, f"layer_summary_{self.method_name}.pdf")
                    fig3.tight_layout()
                    fig3.savefig(p3, dpi=300)
                    plt.close(fig3)

        except Exception as e:
            log.warning(f"Plotting failed: {e}")
