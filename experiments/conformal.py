from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.common import config as load_config
from experiments.interval_core.conformal.split_cp import (  # noqa: E402
    mondrian_conformal_regression,
    proxy_trend_conformal_regression,
    split_conformal_regression,
)
from src.eidos.protocol.strict import five_role_split_by_env, heldout_environment_split  # noqa: E402
from experiments.interval_core.search.engine import EngineAblation, run_eidos_dataset  # noqa: E402


GENERATORS = (
    "feynman_like_blackbody_proxy",
    "feynman_like_kepler",
    "feynman_like_lorentz",
    "feynman_like_oscillator",
    "feynman_like_relativistic",
)


class _NullLedger:
    def log(self, event: dict[str, Any]) -> None:
        _ = event


def _strict_config(lock_permutations: int) -> dict[str, Any]:
    cfg = load_config('conformal')
    cfg['invariance']['lock_permutations'] = int(lock_permutations)
    return cfg


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"sample_id", "env_id", "split", "y_tilde"}
    ]


def _roles(
    frame: pd.DataFrame,
    arm: str,
    seed: int,
    heldout_env: int | float | None = None,
) -> np.ndarray:
    env = frame["env_id"].to_numpy()
    if arm == "exchangeable":
        return five_role_split_by_env(env, seed=seed)
    if arm == "shifted":
        target = np.max(env) if heldout_env is None else heldout_env
        return heldout_environment_split(env, heldout_env=target, seed=seed)
    raise ValueError(f"unknown arm: {arm}")


def _run_cell(
    generator: str,
    arm: str,
    seed: int,
    alpha: float,
    cfg: dict[str, Any],
    heldout_env: int | float | None = None,
    include_proxy_trend: bool = True,
) -> list[dict[str, Any]]:
    task_dir = ROOT / "data" / "conformal" / generator
    frame = pd.read_csv(task_dir / "data.csv")
    if arm == "shifted" and heldout_env is not None:
        # Development backtests emulate the formal ordered shift without
        # allowing later environments into discovery, selection or calibration.
        frame = frame[frame["env_id"].astype(float) <= float(heldout_env)].copy()
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    frame["split"] = _roles(frame, arm, seed, heldout_env=heldout_env)
    frame["sample_id"] = np.arange(frame.shape[0], dtype=int)
    features = _feature_cols(frame)
    result = run_eidos_dataset(
        df=frame,
        dataset_id=generator,
        feature_cols=features,
        target_col="y_tilde",
        env_col="env_id",
        split_col="split",
        delta_y=float(metadata["delta_y"]),
        seed=seed,
        cfg=cfg,
        ledger=_NullLedger(),  # type: ignore[arg-type]
        run_id=f"e6_{generator}_{arm}_{seed}",
        git_hash="nogit",
        ablation=EngineAblation(),
        audit_trace=True,
    )
    model = result.best_model
    if model is None:
        raise RuntimeError(f"EIDOS emitted no model for {generator}")
    cal = frame[frame["split"] == "cal"]
    test = frame[frame["split"] == "test"]
    yhat_cal, _ = model.predict_mu_sigma(cal[features].to_numpy(float))
    yhat_test, _ = model.predict_mu_sigma(test[features].to_numpy(float))
    exchangeable = arm == "exchangeable"
    split_result = split_conformal_regression(
        y_cal=cal["y_tilde"].to_numpy(float),
        yhat_cal=yhat_cal,
        y_test=test["y_tilde"].to_numpy(float),
        yhat_test=yhat_test,
        alpha=alpha,
        exchangeable=exchangeable,
    )
    mondrian_result = mondrian_conformal_regression(
        y_cal=cal["y_tilde"].to_numpy(float),
        yhat_cal=yhat_cal,
        group_cal=cal["env_id"].to_numpy(),
        y_test=test["y_tilde"].to_numpy(float),
        yhat_test=yhat_test,
        group_test=test["env_id"].to_numpy(),
        alpha=alpha,
        exchangeable=exchangeable,
        unseen_group="nearest",
    )
    proxy_result = proxy_trend_conformal_regression(
        y_cal=cal["y_tilde"].to_numpy(float),
        yhat_cal=yhat_cal,
        env_cal=cal["env_id"].to_numpy(),
        y_test=test["y_tilde"].to_numpy(float),
        yhat_test=yhat_test,
        env_test=test["env_id"].to_numpy(),
        alpha=alpha,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    methods = [("split", split_result), ("mondrian", mondrian_result)]
    if include_proxy_trend and arm == "shifted":
        methods.append(("proxy_trend", proxy_result))
    for method, conformal in methods:
        rows.append(
            {
                "generator": generator,
                "arm": arm,
                "method": method,
                "seed": seed,
                "nominal": 1.0 - alpha,
                "observed_coverage": conformal.coverage,
                "mean_interval_length": conformal.avg_length,
                "n_cal": int(cal.shape[0]),
                "n_test": int(test.shape[0]),
                "selected_candidate_id": result.best_candidate_id,
                "guarantee_scope": conformal.guarantee_scope,
                "unseen_group_fallback": "nearest" if arm == "shifted" and method == "mondrian" else "none",
                "shift_heldout_env": float(np.max(test["env_id"])) if arm == "shifted" else np.nan,
            }
        )
    return rows

