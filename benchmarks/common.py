from pathlib import Path
import json
from eidos._io import NullLedger, canonical_bytes, digest, json_safe, write_json

ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return json.loads((ROOT / "configs" / (name + ".json")).read_text(encoding="utf-8"))
