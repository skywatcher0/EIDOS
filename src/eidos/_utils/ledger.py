from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


REQUIRED_LEDGER_FIELDS = [
    "run_id",
    "timestamp",
    "git_hash",
    "seed",
    "dataset_id",
    "env_id",
    "split",
    "delta_y_source",
    "delta_y_value",
    "model_id",
    "candidate_id",
    "dl_struct",
    "dl_param",
    "dl_data",
    "dl_total",
    "inv_test_type",
    "p_inv_work",
    "p_inv_lock",
    "conformal_coverage",
    "conformal_length",
    "failure_reason",
    "notes",
]


class LedgerWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _normalized(self, event: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {k: event.get(k) for k in REQUIRED_LEDGER_FIELDS}
        for k, v in event.items():
            if k not in out:
                out[k] = v
        if not out.get("timestamp"):
            out["timestamp"] = utc_now_iso()
        return out

    def log(self, event: dict[str, Any]) -> None:
        normalized = self._normalized(event)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

