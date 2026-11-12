"""Check freshly computed results and summarize regression quality."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, write_json

SPECS = {
    "benchmark": ("benchmark.csv", ["dataset"], ["rmse", "coverage", "interval_length", "p_inv_lock"], ["candidate_id"]),
    "openml": ("openml.csv", ["dataset", "partition"], ["rmse"], []),
}


def report(out, suites, mode, fingerprint):
    out = Path(out)
    checks, summaries = [], {}
    for suite in suites:
        actual = pd.read_csv(out / suite / "raw.csv")
        errors = int(actual["failed"].fillna(True).astype(bool).sum())
        checks.append({"suite": suite, "check": "successful_rows", "passed": errors == 0 and len(actual) > 0,
                       "failures": errors, "rows": len(actual)})
        file, keys, metrics, categories = SPECS[suite]
        reference = pd.read_csv(ROOT / "reference" / file)
        good = actual[~actual.failed.fillna(True).astype(bool)]
        unique = not actual.duplicated(keys).any()
        actual_keys = set(map(tuple, actual[keys].to_numpy()))
        expected_keys = set(map(tuple, reference[keys].to_numpy()))
        grid_ok = unique and (actual_keys <= expected_keys if mode == "smoke" else actual_keys == expected_keys)
        checks.append({"suite": suite, "check": "task_grid", "passed": grid_ok,
                       "expected": len(reference), "actual": len(actual), "subset": mode == "smoke"})
        if errors or not unique or not len(good):
            continue
        paired = good.merge(reference, on=keys, suffixes=("_actual", "_reference"), how="left", validate="one_to_one", indicator=True)
        checks.append({"suite": suite, "check": "reference_keys", "passed": bool((paired["_merge"] == "both").all())})
        differences = []
        for field in metrics:
            a, b = paired[field + "_actual"].to_numpy(float), paired[field + "_reference"].to_numpy(float)
            strict = np.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)
            atol = 1e-5 if suite == "benchmark" and field in {"rmse", "interval_length"} else 1e-8
            matched = np.isclose(a, b, rtol=1e-5, atol=atol, equal_nan=False)
            checks.append({"suite": suite, "check": field, "passed": bool(matched.all()), "matched": int(matched.sum()),
                           "total": len(matched), "rtol": 1e-5, "atol": atol, "strict_atol": 1e-8,
                           "strict_matched": int(strict.sum()), "max_abs_difference": float(np.max(abs(a - b)))})
            for i in np.flatnonzero(~strict):
                differences.append({**{k: paired.iloc[i][k] for k in keys}, "field": field, "actual": a[i], "reference": b[i]})
        for field in categories:
            a, b = paired[field + "_actual"].astype(str), paired[field + "_reference"].astype(str)
            matched = a == b
            checks.append({"suite": suite, "check": field, "passed": bool(matched.all()), "matched": int(matched.sum()), "total": len(matched)})
            for i in np.flatnonzero(~matched.to_numpy()):
                differences.append({**{k: paired.iloc[i][k] for k in keys}, "field": field, "actual": a.iloc[i], "reference": b.iloc[i]})
        pd.DataFrame(differences, columns=keys + ["field", "actual", "reference"]).to_csv(out / suite / "reference_differences.csv", index=False)
        checks.append({"suite": suite, "check": "record_prediction_roundtrip", "passed": bool(good.decode_exact.all())})
        if suite == "benchmark":
            checks.append({"suite": suite, "check": "replacement_invariance", "passed": bool(good.replacement_identical.all()),
                           "matched": int(good.replacement_identical.sum()), "total": len(good)})
            good = good.merge(pd.read_csv(ROOT / "data/pmlb/catalogue.csv")[["dataset", "task_family", "benchmark_role"]], on="dataset", validate="many_to_one")
            summary = good.groupby("task_family").agg(tasks=("dataset", "size"), median_rmse=("rmse", "median"),
                            median_runtime_sec=("runtime_sec", "median"), median_basis_terms=("final_n_basis", "median"),
                            replacement_passes=("replacement_identical", "sum"))
        else:
            summary = good.groupby("dataset").agg(runs=("rmse", "size"), mean_rmse=("rmse", "mean"), median_rmse=("rmse", "median"))
        summary.to_csv(out / suite / "summary.csv")
        summaries[suite] = summary.reset_index().to_dict("records")
    passed = bool(checks) and all(c["passed"] for c in checks)
    write_json(out / "verification_report.json", {"passed": passed, "scope": mode, "source_config_sha256": fingerprint,
               "note": "Timing is not compared. RMSE uses discovery-standardized quantized responses. Reference data are read only after fitting.", "checks": checks})
    write_json(out / "summary.json", summaries)
    return passed
