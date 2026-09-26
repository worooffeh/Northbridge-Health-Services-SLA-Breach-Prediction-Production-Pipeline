from pathlib import Path
import nbformat

for path in sorted(Path("notebooks").glob("*.ipynb")):
    nbformat.read(path, as_version=4)
    print("valid", path)
