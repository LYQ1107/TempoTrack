"""Pure PyTorch research components for TempoTrack.

The package intentionally does not import MMDetection, MMCV, or the legacy
``masa`` package at import time.  Frozen observations can therefore be
prepared and validated in a small Python environment, while the optional
adapters load the original tracking stack only when explicitly requested.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
