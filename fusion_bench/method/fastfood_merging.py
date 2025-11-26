"""
Backward compatibility module for fastfood_merging -> srp_merging rename.

This module provides backward compatibility for code that imports from
fusion_bench.method.fastfood_merging. All functionality has been moved to
fusion_bench.method.srp_merging.

Please update your imports to use:
    from fusion_bench.method.srp_merging import SRPSubspaceMergeAlgorithm

Legacy imports will continue to work:
    from fusion_bench.method.fastfood_merging import FastfoodSubspaceMergeAlgorithm
"""

import warnings

# Import everything from srp_merging for backward compatibility
from .srp_merging import (
    SRPSubspaceMergeAlgorithm,
    FastfoodSubspaceMergeAlgorithm,
    MultiScaleFastfoodMergeAlgorithm,
    CLIPRatiosOnlyAdaFastfood,
    srp_utils,
    projection_size_estimator,
)

# Warn users about the deprecated module path
warnings.warn(
    "fusion_bench.method.fastfood_merging is deprecated and will be removed in a future version. "
    "Please use fusion_bench.method.srp_merging instead. "
    "FastfoodSubspaceMergeAlgorithm has been renamed to SRPSubspaceMergeAlgorithm "
    "(though FastfoodSubspaceMergeAlgorithm remains as an alias for backward compatibility).",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "SRPSubspaceMergeAlgorithm",
    "FastfoodSubspaceMergeAlgorithm",
    "MultiScaleFastfoodMergeAlgorithm",
    "CLIPRatiosOnlyAdaFastfood",
    "srp_utils",
    "projection_size_estimator",
]
