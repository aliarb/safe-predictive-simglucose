#!/usr/bin/env python3
"""
Quick wrapper for the paper-ready finetuning script.

This repo now uses `examples/rl_finetune_nmpc_for_paper.py` (candidate search)
instead of the step-by-step RL environment in `tune_nmpc_with_rl.py`, because
it is designed to finish reliably and save:
- results/rl_finetuning/best_params.json
- results/rl_finetuning/training_history.csv
- results/rl_finetuning/training_history.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name("rl_finetune_nmpc_for_paper.py")
    cmd = [
        sys.executable,
        str(script),
        "--candidates",
        "15",
        "--tune_hours",
        "6",
        "--final_hours",
        "24",
        "--seed",
        "0",
    ]

    print("=" * 70)
    print("QUICK PAPER-READY FINETUNING (CANDIDATE SEARCH)")
    print("=" * 70)
    print("Running:")
    print("  " + " ".join(cmd))
    print()

    subprocess.check_call(cmd)

    print()
    print("Next (paper comparison):")
    print("  python examples/compare_nmpc_for_paper.py")

