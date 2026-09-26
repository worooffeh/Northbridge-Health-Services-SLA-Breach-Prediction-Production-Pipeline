from __future__ import annotations

import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="/mnt/data")
parser.add_argument("--destination", default="data/raw")
args = parser.parse_args()

source = Path(args.source)
destination = Path(args.destination)
destination.mkdir(parents=True, exist_ok=True)
for name in ["Tickets.csv", "Agents.csv", "Clients.csv"]:
    src = source / name
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, destination / name)
    print("copied", name)
