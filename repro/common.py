from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def config(name):
    return json.loads((ROOT / 'configs' / (name + '.json')).read_text(encoding='utf-8'))

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')

def canonical_bytes(value):
    # No timestamps, host paths, runtime measurements, or test metrics enter this payload.
    return json.dumps(json_safe(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')

def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

class NullLedger:
    def log(self, event):
        pass
