#!/usr/bin/env python
"""Full Test schema10 Dormant Oracle; DIAGNOSTIC_UPPER_BOUND only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempotrack_research.orchestration.v9_oracle import main

if __name__ == "__main__":
    main()
