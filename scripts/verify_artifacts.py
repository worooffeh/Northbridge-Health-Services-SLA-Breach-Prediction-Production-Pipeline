import hashlib, json
from pathlib import Path

model = Path("models/champion/model.joblib")
meta = Path("models/champion/metadata.json")
if not model.exists() or not meta.exists():
    raise SystemExit("Champion artifacts are missing. Run scripts/train_champion.py first.")
metadata = json.loads(meta.read_text())
digest = hashlib.sha256(model.read_bytes()).hexdigest()
print("model_version:", metadata.get("model_version"))
print("expected sha256:", metadata.get("artifact_sha256"))
print("actual sha256:  ", digest)
if metadata.get("artifact_sha256") != digest:
    raise SystemExit("Artifact verification FAILED")
print("Artifact verification PASSED")
