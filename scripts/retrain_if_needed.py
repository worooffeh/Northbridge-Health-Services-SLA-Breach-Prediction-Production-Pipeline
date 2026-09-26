from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from src.monitoring.retrain import retraining_gate

parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true", help="Run training if the gate triggers.")
parser.add_argument("--current-metrics", default=None, help="Optional JSON file containing mature current-period labels/metrics.")
args = parser.parse_args()

meta_path = Path("models/champion/metadata.json")
drift_path = Path("reports/monitoring/drift_report.csv")
metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
drift = pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()
current = json.loads(Path(args.current_metrics).read_text()) if args.current_metrics else None
reference = metadata.get("test_metrics") if current else None
result = retraining_gate(drift, reference, current)
print(json.dumps(result, indent=2))

if result["retrain"] and args.execute:
    subprocess.run([sys.executable, "scripts/train_champion.py"], check=True)
