#!/usr/bin/env python3
"""
DEPRECATED: paper workflow wrapper.

This repo now uses `examples/rl_finetune_nmpc_for_paper.py` (candidate search)
instead of the step-by-step RL environment in `tune_nmpc_with_rl.py`. The new
script is designed to finish and save:
- results/rl_finetuning/best_params.json
- results/rl_finetuning/training_history.csv
- results/rl_finetuning/training_history.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> int:
    script = Path(__file__).with_name("rl_finetune_nmpc_for_paper.py")
    cmd = [
        sys.executable,
        str(script),
        "--candidates",
        "200",
        "--tune_hours",
        "6",
        "--final_hours",
        "24",
        "--seed",
        "0",
    ]

    print("=" * 80)
    print("DEPRECATED: use rl_finetune_nmpc_for_paper.py (paper-ready finetuning)")
    print("=" * 80)
    print("Running:")
    print("  " + " ".join(cmd))
    print()

    subprocess.check_call(cmd)

    print()
    print("Next (paper comparison):")
    print("  python examples/compare_nmpc_for_paper.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

