"""Public training and inference interface for numeric regression tables."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ._io import NullLedger, digest, write_json
from ._utils.data import quantize_y
from .conformal.split_cp import conformal_quantile
from .protocol.strict import ROLE_ORDER, five_role_split_by_env
from .search.engine import run_eidos_dataset
from .serialization import pack, unpack


class EIDOSRegressor:
    """Fit an interpretable regressor while reserving independent evidence roles.

    ``fit`` uses 40% discovery, 15% diagnosis, 15% verification, 15% calibration
    and 15% test rows by default. Only discovery and diagnosis construct the
    model. Pass ``roles`` to supply an existing partition. Predictions and
    intervals are returned in the original target units.
    """

    def __init__(self, *, random_state=42, max_features=8, alpha=0.1):
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**31:
            raise ValueError("random_state must be an integer in [0, 2**31)")
        if isinstance(max_features, bool) or not isinstance(max_features, (int, np.integer)) or max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.random_state = int(random_state)
        self.max_features = int(max_features)
        self.alpha = float(alpha)

    @staticmethod
    def _table(X, names=None):
        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates:
                raise ValueError("Feature names must be unique")
            table = X.copy()
            if not all(isinstance(c, str) for c in table.columns):
                raise ValueError("DataFrame feature names must be strings")
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional numeric table")
            if names is not None and array.shape[1] != len(names):
                raise ValueError(f"Expected {len(names)} feature columns, received {array.shape[1]}")
            table = pd.DataFrame(array, columns=names or [f"feature_{i}" for i in range(array.shape[1])])
        if table.shape[1] == 0 or len(table) == 0:
            raise ValueError("X must contain rows and at least one feature")
        try:
            table = table.astype(float).reset_index(drop=True)
        except (ValueError, TypeError) as error:
            raise ValueError("Features must be numeric; encode categorical columns before fitting") from error
        if not np.isfinite(table.to_numpy()).all():
            raise ValueError("Features must contain finite numbers; handle missing values before fitting")
        return table

    def fit(self, X, y, *, environments=None, roles=None, dataset_id="custom"):
        """Select, refit, verify and calibrate; never refit on the reserved rows."""
        table = self._table(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(table) or not np.isfinite(target).all():
            raise ValueError("y must be a finite one-dimensional array with one value per row")
        if len(table) < 80:
            raise ValueError("At least 80 rows are required for the five evidence roles")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        self.feature_names_in_ = list(table.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        if environments is None:
            # Stable equal-count strata also handle tied or constant first columns.
            ranks = table.iloc[:, 0].rank(method="first").to_numpy() - 1
            env = np.minimum((5 * ranks / len(table)).astype(int), 4)
        else:
            values = np.asarray(environments)
            if values.ndim != 1 or len(values) != len(table) or pd.isna(values).any():
                raise ValueError("environments must contain one non-missing label per row")
            env = pd.factorize(values, sort=False)[0]
        if roles is None:
            partition = five_role_split_by_env(env, seed=self.random_state)
        else:
            partition = np.asarray(roles, dtype=str)
            if partition.ndim != 1 or len(partition) != len(table) or set(partition) != set(ROLE_ORDER):
                raise ValueError("roles must label each row disc, diag, audit, cal or test, with all five roles present")
        disc = partition == "disc"
        if np.count_nonzero(disc) < 12 or np.count_nonzero(partition == "diag") < 3:
            raise ValueError("Provide at least 12 discovery and 3 diagnosis rows")
        usable = [c for c in table if np.std(table.loc[disc, c].to_numpy()) > 1e-12]
        if not usable:
            raise ValueError("No feature varies on the discovery rows")
        if len(usable) > self.max_features:
            correlations = []
            for column in usable:
                value = abs(float(np.corrcoef(table.loc[disc, column], target[disc])[0, 1]))
                correlations.append((value if np.isfinite(value) else 0.0, column))
            usable = [c for _, c in sorted(correlations, reverse=True)[:self.max_features]]
        scaler = StandardScaler().fit(table.loc[disc, usable].to_numpy())
        yc, ys = float(np.mean(target[disc])), float(np.std(target[disc]))
        if ys <= 1e-12:
            raise ValueError("The discovery target must vary")
        features = [f"x{i}" for i in range(len(usable))]
        frame = pd.DataFrame(scaler.transform(table[usable].to_numpy()), columns=features)
        frame["y_tilde"] = quantize_y((target - yc) / ys, 0.001)
        frame["env_id"] = env
        frame["split"] = partition
        frame["sample_id"] = np.arange(len(frame))
        context = {
            "seed": self.random_state, "dataset_id": dataset_id,
            "selected_original_features": usable,
            "feature_affine": {c: [float(scaler.mean_[i]), float(scaler.scale_[i])] for i, c in enumerate(features)},
            "target_affine": [yc, ys], "response_step": 0.001,
        }
        cfg = json.loads(files("eidos").joinpath("defaults.json").read_text(encoding="utf-8"))
        cfg["search"]["feature_affine"] = context["feature_affine"]
        cfg["search"]["target_affine"] = context["target_affine"]
        cfg["conformal"]["alpha_list"] = [self.alpha]
        with threadpool_limits(limits=1):
            result = run_eidos_dataset(
                df=frame, dataset_id=dataset_id, feature_cols=features, target_col="y_tilde",
                env_col="env_id", split_col="split", delta_y=0.001, seed=self.random_state,
                cfg=cfg, ledger=NullLedger(), run_id="fit", git_hash="installed", audit_trace=True,
            )
        self.record_ = pack(result.best_model, float(result.layer_stats["final_param_step"]), result.selection_trace, context)
        self._model = unpack(self.record_)
        self.model_sha256_ = digest(self.record_)
        self.selected_features_ = usable
        self.candidate_id_ = result.best_candidate_id
        self.formula_ = result.best_model.to_expr().to_string()
        self.role_assignments_ = partition.copy()
        self.prepared_data_ = frame
        # Calibrate in raw units, against the original response (not rounded bins).
        calibration = partition == "cal"
        with threadpool_limits(limits=1):
            all_predictions = self.predict(table)
        self.interval_radius_ = conformal_quantile(np.abs(target[calibration] - all_predictions[calibration]), self.alpha)
        test = partition == "test"
        prediction = all_predictions[test]
        truth = target[test]
        self.metrics_ = {
            "test_rmse": float(np.sqrt(np.mean((truth - prediction)**2))),
            "test_mae": float(np.mean(np.abs(truth - prediction))),
            "test_coverage": float(np.mean(np.abs(truth - prediction) <= self.interval_radius_)),
            "interval_width": float(2 * self.interval_radius_),
            "nominal_coverage": 1 - self.alpha,
            "verification_p_value": float(result.p_inv_lock),
            "quantized_target_test_rmse": float(result.rmse_test),
            "basis_terms": len(result.best_model.spec.basis_exprs),
            "role_counts": {r: int(np.sum(partition == r)) for r in ROLE_ORDER},
            "units": "original target units except quantized_target_test_rmse",
        }
        self.test_predictions_ = pd.DataFrame({
            "row_id": np.flatnonzero(test), "target": truth, "prediction": prediction,
            "lower": prediction - self.interval_radius_, "upper": prediction + self.interval_radius_,
        })
        return self

    def _require_fitted(self):
        if not hasattr(self, "record_"):
            raise RuntimeError("Fit the estimator or load a saved model first")

    def predict(self, X):
        """Predict from a numeric DataFrame or an array in original feature order."""
        self._require_fitted()
        table = self._table(X, self.feature_names_in_)
        missing = set(self.selected_features_) - set(table.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        context = self.record_["construction_context"]
        x = table[self.selected_features_].to_numpy(copy=True)
        for i, name in enumerate(self._model.feature_cols):
            center, scale = context["feature_affine"][name]
            x[:, i] = (x[:, i] - center) / scale
        with threadpool_limits(limits=1):
            mean, _ = self._model.predict_mu_sigma(x)
        center, scale = context["target_affine"]
        return mean * scale + center

    def predict_interval(self, X):
        """Return n-by-2 split-conformal bounds at the configured coverage level."""
        prediction = self.predict(X)
        return np.column_stack([prediction - self.interval_radius_, prediction + self.interval_radius_])

    def save(self, path):
        """Write a JSON model; raw data and machine paths are not needed to load it."""
        self._require_fitted()
        write_json(path, {
            "schema": "eidos-model-v1", "record": self.record_, "model_sha256": self.model_sha256_,
            "feature_names_in": self.feature_names_in_, "formula": self.formula_,
            "settings": {"random_state": self.random_state, "max_features": self.max_features, "alpha": self.alpha},
            "calibration": {"radius_hex": float(self.interval_radius_).hex(), "units": "original target"},
        })
        return Path(path)

    @classmethod
    def load(cls, path):
        """Load a data-only JSON record, without executing serialized Python code."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "eidos-model-v1":
            raise ValueError("Unsupported EIDOS model schema")
        record = payload["record"]
        if digest(record) != payload["model_sha256"]:
            raise ValueError("Model record checksum mismatch")
        estimator = cls(**payload["settings"])
        estimator.record_ = record
        estimator._model = unpack(record)
        estimator.model_sha256_ = payload["model_sha256"]
        estimator.feature_names_in_ = payload["feature_names_in"]
        estimator.n_features_in_ = len(estimator.feature_names_in_)
        estimator.selected_features_ = record["construction_context"]["selected_original_features"]
        estimator.candidate_id_ = record["spec"]["candidate_id"]
        estimator.formula_ = payload["formula"]
        estimator.interval_radius_ = float.fromhex(payload["calibration"]["radius_hex"])
        if np.isnan(estimator.interval_radius_) or estimator.interval_radius_ < 0:
            raise ValueError("Invalid calibration radius")
        return estimator
