#!/usr/bin/env python3
from __future__ import annotations

import sys

from prepare_lichess_quiet import main


if __name__ == "__main__":
    # Force mixed defaults unless explicitly overridden by CLI flags.
    if "--position-policy" not in sys.argv:
        sys.argv.extend(["--position-policy", "mixed"])
    if "--mix-all" not in sys.argv:
        sys.argv.extend(["--mix-all", "0.50"])
    if "--mix-tactical" not in sys.argv:
        sys.argv.extend(["--mix-tactical", "0.30"])
    if "--mix-quiet" not in sys.argv:
        sys.argv.extend(["--mix-quiet", "0.20"])
    main()
