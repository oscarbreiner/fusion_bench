from .srp_merging import SRPSubspaceMergeAlgorithm
from .multiscale.multi_scale_fastfood_merging import MultiScaleFastfoodMergeAlgorithm
from .learnable.fastfood_learnable_dim import CLIPRatiosOnlyAdaFastfood
from . import srp_utils
from . import projection_size_estimator

# Backward compatibility alias
FastfoodSubspaceMergeAlgorithm = SRPSubspaceMergeAlgorithm

__all__ = [
    "SRPSubspaceMergeAlgorithm",
    "FastfoodSubspaceMergeAlgorithm",  # backward compatibility
    "MultiScaleFastfoodMergeAlgorithm",
    "CLIPRatiosOnlyAdaFastfood",
    "srp_utils",
    "projection_size_estimator"
]
