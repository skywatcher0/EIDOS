from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .data_code import DataCodeResult


@dataclass
class MdlBreakdown:
    dl_grammar: float
    dl_struct: float
    dl_param: float
    dl_data: float

    @property
    def dl_total(self) -> float:
        return float(self.dl_grammar + self.dl_struct + self.dl_param + self.dl_data)

    def to_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["dl_total"] = self.dl_total
        return out


def build_breakdown(
    dl_grammar: float,
    dl_struct: float,
    dl_param: float,
    data_result: DataCodeResult,
) -> MdlBreakdown:
    return MdlBreakdown(
        dl_grammar=float(dl_grammar),
        dl_struct=float(dl_struct),
        dl_param=float(dl_param),
        dl_data=float(data_result.dl_data_bits),
    )


def save_breakdown(path: str | Path, breakdown: MdlBreakdown) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(breakdown.to_dict(), f, ensure_ascii=False, indent=2)

