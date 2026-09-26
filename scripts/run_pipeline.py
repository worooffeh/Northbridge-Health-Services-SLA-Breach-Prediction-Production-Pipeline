from __future__ import annotations

import argparse
import os
import subprocess
import sys

STEPS = [
    ("audit", [sys.executable, "scripts/run_audit.py"]),
    ("train", [sys.executable, "scripts/train_champion.py"]),
    ("explain", [sys.executable, "scripts/run_explainability.py"]),
    ("monitor", [sys.executable, "scripts/run_monitoring.py"]),
    ("verify", [sys.executable, "scripts/verify_artifacts.py"]),
]

parser = argparse.ArgumentParser(description="Run the Northbridge SLA pipeline end to end.")
parser.add_argument("--skip-explain", action="store_true")
parser.add_argument("--skip-monitor", action="store_true")
args = parser.parse_args()

for name, command in STEPS:
    if name == "explain" and args.skip_explain:
        continue
    if name == "monitor" and args.skip_monitor:
        continue
    print(f"\n=== {name.upper()} ===", flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())

print("\nPipeline completed successfully.")
