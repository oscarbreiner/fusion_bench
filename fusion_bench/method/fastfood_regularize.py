"""
FastFood Regularization Method

Evaluates individual task models with FastFood projection applied to task vectors.
Similar to dummy method but applies FastFood projection on task deltas first.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from copy import deepcopy

import torch
from torch import nn, Tensor
import numpy as np

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.method.fastfood_merging.fastfood_utils import create_projection_ops

log = logging.getLogger(__name__)


class FastfoodRegularizeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """FastFood Regularization via Task Vector Projection"""

    _config_mapping = BaseAlgorithm._config_mapping | {
        "proj_ratio": "proj_ratio",
        "transform_type": "transform_type",
        "device": "device",
        "subspace_scope": "subspace_scope",
        "block_rows": "block_rows",
        "only_project_linear": "only_project_linear",
        "energy_rescale": "energy_rescale",
    }

    def __init__(
        self,
        proj_ratio: float = 0.5,
        use_G: bool = False,  # Deprecated, kept for config compatibility
        device: str = "cuda",
        transform_type: str | None = "srht",    # "fwht" | "srht" | "dct" | "dht" | "none" | None
        subspace_scope: str = "per_tensor",
        block_rows: int = 8192,
        only_project_linear: bool = True,
        energy_rescale: bool = False,  # Deprecated, kept for config compatibility
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.transform_type = str(transform_type)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        self.block_rows = int(block_rows)
        self.only_project_linear = bool(only_project_linear)
        # Energy rescaling removed - projection ops are assumed unitary

        log.info(
            f"FastFood Regularize: proj_ratio={self.proj_ratio}, "
            f"scope={self.subspace_scope}, only_linear={self.only_project_linear}"
        )

    def _get_seed_key(self, param_name: str) -> str:
        if self.subspace_scope == "global":
            return "__GLOBAL__"
        elif self.subspace_scope == "layer":
            parts = param_name.split(".")
            if len(parts) >= 3:
                return ".".join(parts[:3])
            elif len(parts) >= 2:
                return ".".join(parts[:2])
            return param_name
        else:
            return param_name

    def _should_project(self, param_name: str, tensor: Tensor) -> bool:
        if not torch.is_floating_point(tensor):
            return False
        return tensor.ndim == 2 if self.only_project_linear else tensor.ndim >= 1

    @torch.no_grad()
    def _project_matrix(
        self, mat: Tensor, proj_dim: int, seed_key: str, op_cache: Dict
    ) -> Tensor:
        rows, D = mat.shape
        cache_key = (seed_key, D, proj_dim)

        if cache_key not in op_cache:
            fwd, lift = create_projection_ops(
                D, proj_dim, 
                transform_type=self.transform_type,
                seed_key=seed_key, 
                device=self.device
            )
            op_cache[cache_key] = (fwd, lift)
        else:
            fwd, lift = op_cache[cache_key]

        out = torch.empty_like(mat)
        block_size = min(self.block_rows, rows)

        cursor = 0
        while cursor < rows:
            end = min(cursor + block_size, rows)
            block = mat[cursor:end, :]          # stays on self.device
            compressed = fwd(block)
            reconstructed = lift(compressed)
            out[cursor:end, :] = reconstructed
            cursor = end

        return out

    @torch.no_grad()
    def _project_task_vector(
        self, delta_sd: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, float], Dict[str, float]]:
        """
        Returns:
          projected: dict[name->Tensor] projected (or passthrough) deltas ON self.device in fp32
          errors: dict[name->float] relative reconstruction error (L2 norm ratio) for projected tensors
          snr_db: dict[name->float] SNR in dB for projected tensors
        """
        if self.proj_ratio >= 1.0:
            # no projection -> no errors
            return {k: v.detach().to(self.device, dtype=torch.float32) for k, v in delta_sd.items()}, {}, {}

        projected: Dict[str, Tensor] = {}
        errors: Dict[str, float] = {}
        snr_db: Dict[str, float] = {}
        op_cache: Dict = {}

        for param_name, delta_tensor in delta_sd.items():
            # keep computation on self.device and in fp32
            delta = delta_tensor.detach().to(self.device, dtype=torch.float32)

            if not self._should_project(param_name, delta):
                projected[param_name] = delta
                continue

            D = int(delta.shape[-1])
            num_rows = delta.numel() // D
            if num_rows <= 0 or D <= 0:
                projected[param_name] = delta
                continue

            mat = delta.view(num_rows, D)
            proj_dim = max(1, int(D * self.proj_ratio))
            seed_key = self._get_seed_key(param_name)

            mat_reconstructed = self._project_matrix(mat, proj_dim, seed_key, op_cache)
            projected[param_name] = mat_reconstructed.view_as(delta)

            # Calculate reconstruction error and SNR
            original_norm = torch.linalg.norm(mat).item()
            
            if original_norm == 0.0:
                # Zero tensor edge case - no meaningful error
                errors[param_name] = 0.0
                snr_db[param_name] = float('inf')  # Perfect reconstruction of zero
            else:
                noise = mat - mat_reconstructed
                noise_norm = torch.linalg.norm(noise).item()
                
                # Relative error
                errors[param_name] = noise_norm / original_norm
                
                # SNR in dB: 20 * log10(signal_norm / noise_norm)
                if noise_norm == 0.0:
                    snr_db[param_name] = float('inf')  # Perfect reconstruction
                else:
                    snr_db[param_name] = 20.0 * np.log10(original_norm / noise_norm)

        return projected, errors, snr_db

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> nn.Module:
        log.info("=" * 80)
        log.info("FastFood Regularization - Individual Task Evaluation (no merging)")
        log.info("=" * 80)

        if isinstance(modelpool, nn.Module):
            return modelpool

        base_model = modelpool.load_pretrained_or_first_model()
        base_sd = base_model.state_dict()
        task_names = [n for n in modelpool.model_names if n != "_pretrained_"]

        log.info(f"Base model: {len(base_sd)} parameters")
        log.info(f"Tasks: {task_names}")

        all_metrics = {"proj_ratio": self.proj_ratio, "tasks": {}}
        returned_model = None

        for task_name in task_names:
            log.info(f"\nProcessing: {task_name}")

            task_model = modelpool.load_model(task_name)
            task_sd = task_model.state_dict()

            # compute delta only when keys match and are floating
            delta_sd = {
                k: (task_sd[k] - base_sd[k])
                for k in task_sd.keys() & base_sd.keys()
                if torch.is_floating_point(task_sd[k]) and torch.is_floating_point(base_sd[k])
            }

            # project deltas
            delta_proj_sd, recon_errors, snr_values = self._project_task_vector(delta_sd)

            # reconstruct state dict: base + projected_delta (cast to base dtype/device)
            new_sd = {}
            for k in base_sd.keys():
                if k in delta_proj_sd:
                    dp = delta_proj_sd[k].to(device=base_sd[k].device, dtype=base_sd[k].dtype)
                    new_sd[k] = base_sd[k] + dp
                elif k in task_sd:
                    t = task_sd[k]
                    if t.device != base_sd[k].device or t.dtype != base_sd[k].dtype:
                        t = t.to(device=base_sd[k].device, dtype=base_sd[k].dtype)
                    new_sd[k] = t
                else:
                    new_sd[k] = base_sd[k].clone()

            reconstructed_model = deepcopy(base_model)
            reconstructed_model.load_state_dict(new_sd, strict=True)
            returned_model = reconstructed_model  # last one returned; metrics file holds all

            # aggregate errors (only projected tensors)
            layer_errors: Dict[str, list] = {}
            layer_snrs: Dict[str, list] = {}
            for pname, err in recon_errors.items():
                layer_key = self._get_seed_key(pname)
                layer_errors.setdefault(layer_key, []).append(err)
                if pname in snr_values:
                    layer_snrs.setdefault(layer_key, []).append(snr_values[pname])

            avg_layer_errors = {layer: float(np.mean(vals)) for layer, vals in layer_errors.items()}
            avg_layer_snrs = {layer: float(np.mean(vals)) for layer, vals in layer_snrs.items()}
            
            overall_avg_error = float(np.mean(list(recon_errors.values()))) if recon_errors else 0.0
            # Filter out inf values for SNR averaging
            finite_snrs = [v for v in snr_values.values() if not np.isinf(v)]
            overall_avg_snr_db = float(np.mean(finite_snrs)) if finite_snrs else float('inf')

            all_metrics["tasks"][task_name] = {
                "overall_avg_error": overall_avg_error,
                "overall_avg_snr_db": overall_avg_snr_db,
                "avg_layer_errors": avg_layer_errors,
                "avg_layer_snrs_db": avg_layer_snrs,
                "num_projected_params": len(recon_errors),
            }

            log.info(f"  projected_params={len(recon_errors)}  avg_recon_error={overall_avg_error:.6f}  avg_snr={overall_avg_snr_db:.2f} dB")

        self._save_metrics(all_metrics)
        return returned_model if returned_model is not None else base_model

    def _save_metrics(self, metrics: Dict[str, Any]):
        try:
            import hydra
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            with open(output_dir / "reconstruction_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            log.info(f"Saved metrics to: {output_dir}/reconstruction_metrics.json")
        except Exception as e:
            log.warning(f"Could not save metrics: {e}")