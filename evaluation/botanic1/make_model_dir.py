"""Build a harness-ready model directory for one Botanic1 size.

    python make_model_dir.py --size S --out models/

Creates ``models/Botanic1-S/``: symlinks to the Hugging Face snapshot files plus
``ricebench_botanic1.py`` and a ``config.json`` whose ``AutoModel`` entry points to
the adapter. The harness names results after this directory (``Botanic1-S``).
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ADAPTER = Path(__file__).with_name("ricebench_botanic1.py")

parser = argparse.ArgumentParser()
parser.add_argument("--size", required=True, choices=["S", "M", "L", "XL"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--revision", default=None)
args = parser.parse_args()

snapshot = Path(snapshot_download(f"living-models/Botanic1-{args.size}",
                                  revision=args.revision))
model_dir = args.out / f"Botanic1-{args.size}"
model_dir.mkdir(parents=True, exist_ok=True)
for src in snapshot.iterdir():
    dst = model_dir / src.name
    if src.name == "config.json" or dst.exists():
        continue
    dst.symlink_to(src.resolve())

config = json.loads((snapshot / "config.json").read_text())
config["auto_map"]["AutoModel"] = "ricebench_botanic1.Botanic1ForRiceBenchmark"
(model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
shutil.copy(ADAPTER, model_dir / ADAPTER.name)
(model_dir / "SOURCE").write_text(
    f"living-models/Botanic1-{args.size} @ {snapshot.name}\n"
)
print(model_dir, snapshot.name)
