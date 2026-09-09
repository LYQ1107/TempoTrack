"""Compatibility shim for the archived VOVTrack/COVTrack runners.

The upstream runners were released against NumPy versions that still exposed
the deprecated scalar aliases.  Keep the compatibility change outside the
upstream repositories and load it only for their reproduction processes.
"""

import numpy as _np


if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float
