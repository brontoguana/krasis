"""Centralized timing flags for Krasis — zero-overhead when disabled.

Usage:
    from krasis.timing import TIMING

    # In hot paths, check the boolean directly (no function call overhead):
    if TIMING.decode:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # Toggle at runtime (e.g. from meta-optimiser):
    TIMING.decode = True   # enable decode timing
    TIMING.decode = False  # disable (zero overhead in hot path)

    # Or toggle from env var on first import:
    #   KRASIS_DECODE_TIMING=1 python ...
"""

import os


class TimingFlags:
    """Simple boolean flags — attribute access is a single dict lookup."""
    __slots__ = ("decode", "prefill", "diag")

    def __init__(self):
        self.decode: bool = os.environ.get("KRASIS_DECODE_TIMING", "") == "1"
        self.prefill: bool = os.environ.get("KRASIS_PREFILL_TIMING", "") == "1"
        self.diag: bool = os.environ.get("KRASIS_DIAG", "") == "1"


# Singleton — import this from anywhere
TIMING = TimingFlags()
