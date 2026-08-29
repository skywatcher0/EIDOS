from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from table5_experiment import (  # noqa: E402
    NOISE_SIGMA,
    REGISTERED_SEEDS,
    run_table5_experiment,
)


GENERATED_FILES = (
    "table5_eidos.csv",
    "table5_eidos_all_seeds.csv",
    "table5_five_seed_gate.csv",
    "candidate_bank.csv",
    "verification_report.json",
    "environment.json",
)


def _environment() -> dict[str, object]:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "PyYAML"]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "noise_sigma": NOISE_SIGMA,
        "registered_seeds": list(REGISTERED_SEEDS),
    }


def _verify(table5: pd.DataFrame, reference_path: Path) -> dict[str, object]:
    expected = pd.read_csv(reference_path)
    exact_columns = [
        "case",
        "generator",
        "seed",
        "truth_formula",
        "eidos_formula",
        "selected_candidate_id",
    ]
    numeric_columns = ["noise_sigma", "semantic_nrmse"]
    if list(table5["generator"]) != list(expected["generator"]):
        return {"passed": False, "reason": "case order or membership differs"}

    details: list[dict[str, object]] = []
    passed = True
    for actual_row, expected_row in zip(
        table5.to_dict(orient="records"),
        expected.to_dict(orient="records"),
        strict=True,
    ):
        mismatches: list[dict[str, object]] = []
        for column in exact_columns:
            if actual_row[column] != expected_row[column]:
                mismatches.append(
                    {
                        "field": column,
                        "actual": actual_row[column],
                        "reference": expected_row[column],
                    }
                )
        for column in numeric_columns:
            if not np.isclose(
                float(actual_row[column]),
                float(expected_row[column]),
                rtol=0.0,
                atol=5.0e-7,
            ):
                mismatches.append(
                    {
                        "field": column,
                        "actual": actual_row[column],
                        "reference": expected_row[column],
                    }
                )
        row_ok = not mismatches
        passed = passed and row_ok
        details.append(
            {
                "generator": actual_row["generator"],
                "passed": row_ok,
                "actual_semantic_nrmse": actual_row["semantic_nrmse"],
                "reference_semantic_nrmse": expected_row["semantic_nrmse"],
                "mismatches": mismatches,
            }
        )
    return {
        "passed": passed,
        "reference": str(reference_path.relative_to(ROOT)),
        "absolute_tolerance": 5.0e-7,
        "rows": details,
    }


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {path}. "
            "Choose another --output-dir or pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the EIDOS records and five-seed gate reported in Table 5."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir, bool(args.overwrite))
    started = time.perf_counter()
    table5, all_runs, gate, bank = run_table5_experiment()

    table5.to_csv(output_dir / "table5_eidos.csv", index=False)
    all_runs.to_csv(output_dir / "table5_eidos_all_seeds.csv", index=False)
    gate.to_csv(output_dir / "table5_five_seed_gate.csv", index=False)
    bank.to_csv(output_dir / "candidate_bank.csv", index=False)

    report = _verify(table5, ROOT / "reference" / "table5_eidos_reference.csv")
    report["five_seed_gate_passed"] = bool(
        (gate["recovered"] == gate["registered_runs"]).all()
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["generated_files"] = list(GENERATED_FILES)
    report["passed"] = bool(report["passed"] and report["five_seed_gate_passed"])

    (output_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    shown = table5[["case", "eidos_formula", "semantic_nrmse"]].copy()
    shown["semantic_nrmse"] = shown["semantic_nrmse"].map(lambda value: f"{value:.6f}")
    print("\nTable 5 - EIDOS rows\n")
    print(shown.to_string(index=False))
    print("\nFive-seed recovery gate\n")
    print(gate[["case", "gate"]].to_string(index=False))
    print(f"\nVerification: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Results: {output_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
