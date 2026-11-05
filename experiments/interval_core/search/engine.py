from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from experiments.interval_core.canon.semhash import semantic_hash
from experiments.interval_core.canon.syn import canonicalize_syn
from experiments.interval_core.conformal.split_cp import ConformalResult, conformal_quantile, split_conformal_regression
from experiments.interval_core.dsl.grammar import Const, Expr, Var, op1, op2, op3
from experiments.interval_core.dsl.domain_safety import interval_analysis
from experiments.interval_core.dsl.types_units import UNIT_DIMENSIONLESS, infer_type_unit
from experiments.interval_core.invariance.crossfit import crossfit_residuals
from experiments.interval_core.invariance.mmd_rff import mmd_rff_permutation_test
from experiments.interval_core.invariance.residual_normalization import normalize_residuals_by_env
from experiments.interval_core.mdl.data_code import dl_data_bin_likelihood, dl_data_naive_pdf
from experiments.interval_core.mdl.fisher_code import (
    estimate_fisher_info_with_diagnostics,
    fisher_param_code_bits,
    uniform_param_code_bits,
)
from experiments.interval_core.mdl.param_code import quantize_parameters
from experiments.interval_core.mdl.struct_code import grammar_code_bits, grammar_extension_code_bits, struct_code_bits
from experiments.interval_core.mdl.total import build_breakdown
from experiments.interval_core.search.models import (
    CandidateSpec,
    LinearBasisModel,
    adaptive_trig_candidate_library,
    default_candidate_library,
    physics_template_candidate_library,
    structural_boost_candidate_library,
)
from src.utils.ledger import LedgerWriter


@dataclass
class EngineAblation:
    use_bin_likelihood: bool = True
    use_fisher: bool = True
    use_crossfit: bool = True
    use_unbiased_scaling: bool = True
    conformal_in_inner_loop: bool = False


@dataclass
class EngineResult:
    best_candidate_id: str
    rmse_test: float
    nll_test: float
    dl_total: float
    p_inv_work: float
    p_inv_lock: float
    conformal_coverage: float
    conformal_length: float
    topk: list[dict[str, Any]]
    runtime_sec: float
    layer_stats: dict[str, Any]
    best_model: LinearBasisModel | None = None
    topk_feature_bank: list[dict[str, Any]] | None = None
    test_sample_ids: np.ndarray | None = None
    test_y: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    test_lower: np.ndarray | None = None
    test_upper: np.ndarray | None = None


@dataclass
class CandidateScreenReport:
    accepted: list[CandidateSpec]
    rejected: list[dict[str, Any]]
    n_before: int
    n_after: int


@dataclass
class CandidateCertificate:
    p_inv_lock: float
    conformal_coverage: float
    conformal_length: float


def _concept_bank_gate_status(
    delta_dl_parent_bits: float,
    candidate_cert: CandidateCertificate,
    parent_cert: CandidateCertificate,
    alpha_inv: float,
    eps_dl: float,
    bonus_bits: float,
    cov_tol: float,
    len_rel_tol: float,
) -> dict[str, Any]:
    min_delta = max(0.0, float(eps_dl) - float(bonus_bits))
    dl_ok = bool(np.isfinite(delta_dl_parent_bits) and (float(delta_dl_parent_bits) >= min_delta))
    lock_ok = bool(np.isfinite(candidate_cert.p_inv_lock) and (float(candidate_cert.p_inv_lock) >= float(alpha_inv)))
    cov_ok = bool(
        np.isfinite(candidate_cert.conformal_coverage)
        and np.isfinite(parent_cert.conformal_coverage)
        and (float(candidate_cert.conformal_coverage) >= float(parent_cert.conformal_coverage) - float(cov_tol))
    )
    len_ok = bool(
        np.isfinite(candidate_cert.conformal_length)
        and np.isfinite(parent_cert.conformal_length)
        and (float(candidate_cert.conformal_length) <= float(parent_cert.conformal_length) * (1.0 + float(len_rel_tol)))
    )
    reasons: list[str] = []
    if not dl_ok:
        reasons.append("dl_not_improved")
    if not lock_ok:
        reasons.append("lock_invariance_fail")
    if not cov_ok:
        reasons.append("conformal_coverage_drop")
    if not len_ok:
        reasons.append("conformal_length_increase")
    passed = dl_ok and lock_ok and cov_ok and len_ok
    return {
        "passed": bool(passed),
        "dl_ok": bool(dl_ok),
        "lock_ok": bool(lock_ok),
        "cov_ok": bool(cov_ok),
        "len_ok": bool(len_ok),
        "failure_reason": None if passed else "|".join(reasons) or "concept_rollback",
    }


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _pooled_out_of_sample_rmse(
    crossfit_rmse: np.ndarray | pd.Series,
    diagnosis_rmse: np.ndarray | pd.Series,
    n_crossfit: int,
    n_diagnosis: int,
) -> np.ndarray:
    """Pool two disjoint out-of-sample RMSE estimates by their sample counts."""

    if n_crossfit <= 0 or n_diagnosis <= 0:
        raise ValueError("pooled RMSE requires positive sample counts")
    cv = np.asarray(crossfit_rmse, dtype=float)
    diag = np.asarray(diagnosis_rmse, dtype=float)
    if cv.shape != diag.shape:
        raise ValueError("crossfit and diagnosis RMSE arrays must align")
    return np.sqrt((n_crossfit * cv**2 + n_diagnosis * diag**2) / (n_crossfit + n_diagnosis))


def _nll_student(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, df: float = 3.0) -> float:
    from scipy.stats import t

    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    ll = t.logpdf(z, df=df) - np.log(sigma)
    return float(-np.mean(ll))


def _stable_seed(base_seed: int, key: str, salt: int = 0) -> int:
    h = zlib.crc32(f"{key}|{salt}".encode("utf-8")) & 0xFFFFFFFF
    return int((int(base_seed) + h) % (2**32 - 1))


def _format_unit_signature(unit: dict[str, float]) -> str:
    if not unit:
        return "dimensionless"
    items = sorted((str(k), float(v)) for k, v in unit.items() if abs(float(v)) > 1e-12)
    if not items:
        return "dimensionless"
    toks = []
    for k, v in items:
        if abs(v - round(v)) < 1e-12:
            toks.append(f"{k}^{int(round(v))}")
        else:
            toks.append(f"{k}^{v:.6g}")
    return "*".join(toks)


def _primary_conformal_alpha(conf_cfg: dict[str, Any]) -> float:
    alpha_list = conf_cfg.get("alpha_list", [0.1])
    if isinstance(alpha_list, list) and alpha_list:
        if len(alpha_list) > 1:
            return float(alpha_list[1])
        return float(alpha_list[0])
    return 0.1


def _evaluate_lock_invariance(
    model: LinearBasisModel,
    df_lock: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    env_col: str,
    inv_cfg: dict[str, Any],
    seed: int,
) -> float:
    return float(
        _evaluate_lock_invariance_result(
            model=model,
            df_lock=df_lock,
            feature_cols=feature_cols,
            target_col=target_col,
            env_col=env_col,
            inv_cfg=inv_cfg,
            seed=seed,
        ).p_value
    )


def _evaluate_lock_invariance_result(
    model: LinearBasisModel,
    df_lock: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    env_col: str,
    inv_cfg: dict[str, Any],
    seed: int,
) -> float:
    if df_lock.empty:
        return mmd_rff_permutation_test(
            z=np.zeros(2, dtype=float),
            env=np.zeros(2, dtype=int),
            n_features=int(inv_cfg.get("rff_features", 128)),
            n_perm=1,
            seed=seed,
        )
    mu_lock, sigma_lock = model.predict_mu_sigma(df_lock[feature_cols].to_numpy(float))
    z_lock = (df_lock[target_col].to_numpy(float) - mu_lock) / np.maximum(sigma_lock, 1e-8)
    z_lock = normalize_residuals_by_env(
        z_lock,
        df_lock[env_col].to_numpy(int),
        mode=str(inv_cfg.get("residual_normalization", "global")),
        min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
        shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
    )
    return mmd_rff_permutation_test(
        z=z_lock,
        env=df_lock[env_col].to_numpy(int),
        n_features=int(inv_cfg.get("rff_features", 128)),
        n_perm=int(inv_cfg.get("lock_permutations", 500)),
        seed=seed,
    )


def _evaluate_conformal_result(
    model: LinearBasisModel,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float,
) -> ConformalResult:
    if df_cal.empty or df_test.empty:
        raise ValueError("calibration and test roles must both be non-empty")
    x_cal = df_cal[feature_cols].to_numpy(float)
    x_test = df_test[feature_cols].to_numpy(float)
    y_cal = df_cal[target_col].to_numpy(float)
    y_test = df_test[target_col].to_numpy(float)
    mu_cal, sigma_cal = model.predict_mu_sigma(x_cal)
    mu_test, sigma_test = model.predict_mu_sigma(x_test)

    sigma_eps = 1e-6
    sigma_rel_tol = 5e-2
    sigma_cal = np.asarray(sigma_cal, dtype=float)
    sigma_test = np.asarray(sigma_test, dtype=float)
    sigma_ok = (
        sigma_cal.shape == y_cal.shape
        and sigma_test.shape == y_test.shape
        and bool(np.isfinite(sigma_cal).all())
        and bool(np.isfinite(sigma_test).all())
        and bool((sigma_cal > sigma_eps).all())
        and bool((sigma_test > sigma_eps).all())
    )
    if sigma_ok:
        sigma_center = float(np.mean(np.abs(sigma_cal)))
        sigma_rel_var = float(np.std(sigma_cal) / max(sigma_center, sigma_eps))
        sigma_ok = bool(sigma_rel_var >= sigma_rel_tol)

    if sigma_ok:
        scores = np.abs(y_cal - mu_cal) / np.maximum(sigma_cal, sigma_eps)
        if bool(np.isfinite(scores).all()):
            q_hat = conformal_quantile(scores, alpha=alpha)
            if np.isfinite(q_hat):
                radius = q_hat * np.maximum(sigma_test, sigma_eps)
                lo = mu_test - radius
                hi = mu_test + radius
                inside = (y_test >= lo) & (y_test <= hi)
                lengths = hi - lo
                if bool(np.isfinite(lengths).all()):
                    return ConformalResult(
                        alpha=float(alpha),
                        q_hat=float(q_hat),
                        coverage=float(np.mean(inside)),
                        avg_length=float(np.mean(lengths)),
                        p10_length=float(np.quantile(lengths, 0.1)),
                        p90_length=float(np.quantile(lengths, 0.9)),
                        lower=np.asarray(lo, dtype=float).copy(),
                        upper=np.asarray(hi, dtype=float).copy(),
                    )

    return split_conformal_regression(
        y_cal=y_cal,
        yhat_cal=mu_cal,
        y_test=y_test,
        yhat_test=mu_test,
        alpha=alpha,
    )


def _evaluate_conformal_certificate(
    model: LinearBasisModel,
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float,
) -> tuple[float, float]:
    result = _evaluate_conformal_result(
        model=model,
        df_cal=df_cal,
        df_test=df_test,
        feature_cols=feature_cols,
        target_col=target_col,
        alpha=alpha,
    )
    return float(result.coverage), float(result.avg_length)


def _stratified_equal_subsample(
    df: pd.DataFrame, env_col: str, n_per_env: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx_all: list[int] = []
    for env_id, part in df.groupby(env_col):
        idx = part.index.to_numpy()
        if idx.size <= n_per_env:
            idx_all.extend(idx.tolist())
        else:
            sel = rng.choice(idx, size=n_per_env, replace=False)
            idx_all.extend(sel.tolist())
    return df.loc[sorted(idx_all)].copy()


def _scaled_sub_dl(
    full_work: pd.DataFrame,
    sub_work: pd.DataFrame,
    bits_sub_per_sample: np.ndarray,
    env_col: str,
    use_unbiased_scaling: bool,
) -> float:
    sub = sub_work.copy()
    sub["bits_sub"] = bits_sub_per_sample
    total = 0.0
    for env_id, p_sub in sub.groupby(env_col):
        n_sub = p_sub.shape[0]
        n_work = int((full_work[env_col] == env_id).sum())
        scale = (n_work / max(1, n_sub)) if use_unbiased_scaling else 1.0
        total += float(scale * p_sub["bits_sub"].sum())
    return total


def _priority_features_by_correlation(
    df_work: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_n: int,
) -> list[str]:
    if top_n <= 0:
        return []
    y = df_work[target_col].to_numpy(float)
    out: list[tuple[float, str]] = []
    for c in feature_cols:
        x = df_work[c].to_numpy(float)
        x_std = float(np.std(x))
        if not np.isfinite(x_std) or x_std < 1e-12:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        corr_abs = float(np.abs(corr)) if np.isfinite(corr) else 0.0
        out.append((corr_abs, c))
    out.sort(key=lambda kv: (-kv[0], kv[1]))
    ranked = [c for _, c in out]
    if not ranked:
        return feature_cols[: min(len(feature_cols), top_n)]
    return ranked[: min(len(ranked), top_n)]


def _prune_collinear_features(
    df_work: pd.DataFrame,
    candidate_features: list[str],
    threshold: float,
) -> list[str]:
    if not candidate_features:
        return []
    thr = float(np.clip(threshold, 0.0, 0.999999))
    keep: list[str] = []
    for f in candidate_features:
        x_f = df_work[f].to_numpy(float)
        if np.std(x_f) < 1e-12:
            continue
        is_collinear = False
        for k in keep:
            x_k = df_work[k].to_numpy(float)
            if np.std(x_k) < 1e-12:
                continue
            corr = np.corrcoef(x_f, x_k)[0, 1]
            corr_abs = float(np.abs(corr)) if np.isfinite(corr) else 0.0
            if corr_abs >= thr:
                is_collinear = True
                break
        if not is_collinear:
            keep.append(f)
    return keep if keep else candidate_features[:1]


def _and_expr(terms: list[Expr]) -> Expr:
    if not terms:
        return Const(1.0)
    out = terms[0]
    for term in terms[1:]:
        out = op2("mul", out, term)
    return out


def _soft_step(expr: Expr, sharpness: float = 12.0) -> Expr:
    z = op1("neg", op2("mul", Const(float(sharpness)), expr))
    z = op3("clip", z, Const(-20.0), Const(20.0))
    return op2("div", Const(1.0), op2("add", Const(1.0), op1("exp", z)))


def _tree_leaf_indicator_exprs(tree: Any, feature_cols: list[str], max_terms: int) -> list[Expr]:
    tree_ = tree.tree_
    out: list[Expr] = []

    def walk(node: int, path: list[Expr]) -> None:
        if len(out) >= max_terms:
            return
        left = int(tree_.children_left[node])
        right = int(tree_.children_right[node])
        if left == right:
            out.append(_and_expr(path))
            return
        feat_idx = int(tree_.feature[node])
        if feat_idx < 0 or feat_idx >= len(feature_cols):
            return
        feat = feature_cols[feat_idx]
        thr = float(tree_.threshold[node])
        left_cond = op3("if", op2("sub", Const(thr), Var(feat)), Const(1.0), Const(0.0))
        right_cond = op3("if", op2("sub", Var(feat), Const(thr)), Const(1.0), Const(0.0))
        walk(left, path + [left_cond])
        walk(right, path + [right_cond])

    walk(0, [])
    return out[:max_terms]


def _partition_candidate_library(
    df_work: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    search_cfg: dict[str, Any],
    seed: int,
) -> list[CandidateSpec]:
    if not bool(search_cfg.get("enable_partition_candidates", False)):
        return []
    try:
        from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
    except Exception:
        return []

    x = df_work[feature_cols].to_numpy(float)
    y = df_work[target_col].to_numpy(float)
    if x.shape[0] < 12 or x.shape[1] == 0:
        return []

    max_terms = max(4, int(search_cfg.get("partition_max_terms", 180)))
    raw_depths = search_cfg.get("partition_tree_depths", [search_cfg.get("partition_tree_depth", 5)])
    if not isinstance(raw_depths, list):
        raw_depths = [raw_depths]
    depths = [max(2, int(v)) for v in raw_depths]
    raw_min_leaf = search_cfg.get("partition_min_leaf_grid", [search_cfg.get("partition_min_samples_leaf", 2)])
    if not isinstance(raw_min_leaf, list):
        raw_min_leaf = [raw_min_leaf]
    min_leaf_values = [max(1, int(v)) for v in raw_min_leaf]
    raw_ridges = search_cfg.get("partition_ridge_grid", [search_cfg.get("partition_ridge", 0.06)])
    if not isinstance(raw_ridges, list):
        raw_ridges = [raw_ridges]
    ridge_values = [float(v) for v in raw_ridges]

    out: list[CandidateSpec] = []
    first_tree = True
    for depth in depths:
        for min_leaf in min_leaf_values:
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed + 101 * depth + 17 * min_leaf)
            tree.fit(x, y)
            tree_terms = _tree_leaf_indicator_exprs(tree, feature_cols, max_terms=max_terms)
            if not tree_terms:
                continue
            for ridx, ridge in enumerate(ridge_values):
                out.append(
                    CandidateSpec(
                        candidate_id="struct_partition_tree"
                        if first_tree and ridx == 0
                        else f"struct_partition_tree_d{depth}_l{min_leaf}_r{ridx}",
                        basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols] + tree_terms,
                        sigma_mode="constant",
                        ridge=float(ridge),
                    )
                )
            first_tree = False
    if not out:
        return []

    depth = depths[min(1, len(depths) - 1)]
    min_leaf = min_leaf_values[0]
    ridge = ridge_values[min(1, len(ridge_values) - 1)]

    forest_terms: list[Expr] = []
    n_trees = max(1, int(search_cfg.get("partition_forest_trees", 10)))
    per_tree_terms = max(2, int(np.ceil(max_terms / n_trees)))
    for j in range(n_trees):
        if len(forest_terms) >= max_terms:
            break
        extra = ExtraTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed + 7919 * (j + 1))
        extra.fit(x, y)
        forest_terms.extend(_tree_leaf_indicator_exprs(extra, feature_cols, max_terms=per_tree_terms))
    if forest_terms:
        out.append(
            CandidateSpec(
                candidate_id="struct_partition_forest",
                basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols] + forest_terms[:max_terms],
                sigma_mode="constant",
                ridge=float(search_cfg.get("partition_forest_ridge", ridge)),
            )
        )
    return out


def _topk_diverse(
    ranked_df: pd.DataFrame,
    top_k: int,
    df_work: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    if ranked_df.empty or top_k <= 0:
        return ranked_df.head(0)
    n_probe = min(128, df_work.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(df_work.shape[0]), size=n_probe, replace=False)
    x_probe = df_work.iloc[idx][feature_cols].to_numpy(float)

    selected_idx: list[int] = []
    seen_sem: set[tuple[float, ...]] = set()
    for ridx, row in ranked_df.iterrows():
        model = row["model"]
        mu, _ = model.predict_mu_sigma(x_probe)
        med = float(np.median(mu))
        mad = float(np.median(np.abs(mu - med)) + 1e-6)
        z = np.clip((mu - med) / mad, -8.0, 8.0)
        sem = tuple(np.round(z, 2).tolist())
        if sem in seen_sem:
            continue
        seen_sem.add(sem)
        selected_idx.append(ridx)
        if len(selected_idx) >= top_k:
            break

    if len(selected_idx) < top_k:
        for ridx in ranked_df.index:
            if ridx in selected_idx:
                continue
            selected_idx.append(ridx)
            if len(selected_idx) >= top_k:
                break
    return ranked_df.loc[selected_idx]


def _canonicalize_spec(spec: CandidateSpec) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=spec.candidate_id,
        basis_exprs=[canonicalize_syn(e) for e in spec.basis_exprs],
        sigma_mode=spec.sigma_mode,
        noise=spec.noise,
        df=spec.df,
        ridge=spec.ridge,
    )


def _screen_candidates(
    candidates: list[CandidateSpec],
    df_work: pd.DataFrame,
    feature_cols: list[str],
    clip_ratio_threshold: float,
    seed: int,
    forced_candidate_id: str | None = None,
    var_units: dict[str, dict[str, float]] | None = None,
) -> CandidateScreenReport:
    if not candidates:
        return CandidateScreenReport(accepted=[], rejected=[], n_before=0, n_after=0)

    var_intervals = {
        c: (
            float(np.nanmin(df_work[c].to_numpy(float))),
            float(np.nanmax(df_work[c].to_numpy(float))),
        )
        for c in feature_cols
    }
    var_types = {c: "Real" for c in feature_cols}
    unit_map = {c: dict(UNIT_DIMENSIONLESS) for c in feature_cols}
    if isinstance(var_units, dict):
        for c in feature_cols:
            if c in var_units and isinstance(var_units[c], dict):
                unit_map[c] = {str(k): float(v) for k, v in var_units[c].items()}

    rng = np.random.default_rng(seed)
    n_probe = min(256, df_work.shape[0])
    probe_idx = rng.choice(np.arange(df_work.shape[0]), size=n_probe, replace=False)
    probe = df_work.iloc[probe_idx]
    probe_map = {c: probe[c].to_numpy(float) for c in feature_cols}

    accepted: list[CandidateSpec] = []
    rejected: list[dict[str, Any]] = []
    seen_syn: set[tuple[str, ...]] = set()
    seen_sem: set[tuple[str, ...]] = set()

    for spec in candidates:
        c_spec = _canonicalize_spec(spec)
        sem_tokens: list[str] = []
        max_clip_ratio = 0.0
        fail_reason: str | None = None

        for expr in c_spec.basis_exprs:
            tu = infer_type_unit(expr, var_types=var_types, var_units=unit_map)
            if not tu.valid:
                fail_reason = tu.reason or "type_mismatch"
                break
            iv = interval_analysis(expr, var_intervals=var_intervals)
            if not iv.valid and iv.definitely_invalid:
                fail_reason = "domain_error"
                break

            sem = semantic_hash(expr, probes=probe_map, tol=5e-2)
            if sem.failure_reason is not None:
                fail_reason = sem.failure_reason
                break
            sem_tokens.append(sem.sem_hash)
            max_clip_ratio = max(max_clip_ratio, float(sem.clip_ratio))

        if fail_reason is None and max_clip_ratio > clip_ratio_threshold:
            fail_reason = "clip_ratio_high"

        syn_key = tuple(e.to_string() for e in c_spec.basis_exprs)
        sem_key = tuple(sem_tokens)
        if fail_reason is None and c_spec.candidate_id != forced_candidate_id:
            if syn_key in seen_syn:
                fail_reason = "canonical_duplicate_syn"
            elif sem_key in seen_sem:
                fail_reason = "canonical_duplicate_sem"

        if fail_reason is not None:
            rejected.append(
                {
                    "candidate_id": c_spec.candidate_id,
                    "failure_reason": fail_reason,
                    "clip_ratio_max": max_clip_ratio,
                }
            )
            continue

        accepted.append(c_spec)
        seen_syn.add(syn_key)
        seen_sem.add(sem_key)

    return CandidateScreenReport(
        accepted=accepted,
        rejected=rejected,
        n_before=len(candidates),
        n_after=len(accepted),
    )


def _mine_reusable_concepts(
    specs: list[CandidateSpec],
    max_new_concepts: int,
    min_reuse: int = 2,
    var_units: dict[str, dict[str, float]] | None = None,
) -> list[Any]:
    if max_new_concepts <= 0:
        return []
    unit_map = var_units or {}
    freq: dict[str, int] = {}
    expr_map: dict[str, Any] = {}
    for spec in specs:
        for expr in spec.basis_exprs:
            cexpr = canonicalize_syn(expr)
            if cexpr.op == "const":
                continue
            if cexpr.op == "var":
                vname = str(cexpr.name or "")
                vunits = unit_map.get(vname, {})
                # Keep unit-bearing variables as eligible reusable concepts.
                if not isinstance(vunits, dict) or (len(vunits) == 0):
                    continue
            key = cexpr.to_string()
            freq[key] = freq.get(key, 0) + 1
            expr_map[key] = cexpr

    ranked = sorted(
        [
            (k, v)
            for k, v in freq.items()
            if (
                (v >= min_reuse)
                or (
                    expr_map[k].op == "var"
                    and isinstance(unit_map.get(str(expr_map[k].name or ""), {}), dict)
                    and len(unit_map.get(str(expr_map[k].name or ""), {})) > 0
                    and v >= 1
                )
            )
        ],
        key=lambda kv: (-kv[1], len(kv[0])),
    )
    out = [expr_map[k] for k, _ in ranked[:max_new_concepts]]
    return out


def _build_concept_augmented_specs(
    base_specs: list[CandidateSpec],
    concepts: list[Any],
    max_augmented: int,
) -> list[CandidateSpec]:
    if (not concepts) or max_augmented <= 0:
        return []
    out: list[CandidateSpec] = []
    cap = max(1, max_augmented // max(1, len(concepts)))
    for i, spec in enumerate(base_specs[:cap]):
        existing = {e.to_string() for e in spec.basis_exprs}
        for j, concept_expr in enumerate(concepts):
            c_str = concept_expr.to_string()
            if c_str in existing:
                continue
            aug = CandidateSpec(
                candidate_id=f"{spec.candidate_id}__cg{i+1}_{j+1}",
                basis_exprs=list(spec.basis_exprs) + [concept_expr],
                sigma_mode=spec.sigma_mode,
                noise=spec.noise,
                df=spec.df,
                ridge=spec.ridge,
                parent_candidate_id=spec.candidate_id,
            )
            out.append(aug)
            if len(out) >= max_augmented:
                return out
    return out


def _fit_score_single(
    spec: CandidateSpec,
    df_fit: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    delta_y: float,
    use_bin_likelihood: bool,
    use_fisher: bool,
    fisher_probe_points: int,
    fisher_jitter: float,
    robust_eps: float,
    device: str,
    max_fisher_params: int,
    fisher_seed: int,
    new_concepts: int = 0,
    strict_param_code: bool = False,
    fixed_param_step: float = 1e-3,
) -> dict[str, Any]:
    x_fit = df_fit[feature_cols].to_numpy(float)
    y_fit = df_fit[target_col].to_numpy(float)
    x_eval = df_eval[feature_cols].to_numpy(float)
    y_eval = df_eval[target_col].to_numpy(float)

    model = LinearBasisModel(spec=spec, feature_cols=feature_cols, ridge=spec.ridge, device=device)
    model.fit(x_fit, y_fit)
    quantized_param_bits = float("nan")
    if strict_param_code:
        quantized = quantize_parameters(
            model.param_vector(),
            steps=float(fixed_param_step),
        )
        model.set_param_vector(quantized.values)
        quantized_param_bits = quantized.code_bits
    mu_eval, sigma_eval = model.predict_mu_sigma(x_eval)
    if use_bin_likelihood:
        dres = dl_data_bin_likelihood(
            y_tilde=y_eval,
            mu=mu_eval,
            sigma=sigma_eval,
            delta_y=delta_y,
            noise="student_t",
            df=spec.df,
            robust_eps=robust_eps,
        )
    else:
        dres = dl_data_naive_pdf(y_tilde=y_eval, mu=mu_eval, sigma=sigma_eval, noise="student_t", df=spec.df)

    expr = model.to_expr()
    if strict_param_code:
        dl_struct = grammar_extension_code_bits(list(spec.basis_exprs))
        dl_grammar = 0.0
    else:
        dl_struct = struct_code_bits(expr).dl_struct_bits
        dl_grammar = grammar_code_bits(new_concepts=new_concepts)

    if use_fisher and model.n_params <= max_fisher_params:
        rng = np.random.default_rng(fisher_seed)
        probe_count = min(fisher_probe_points, x_fit.shape[0])
        probe_idx = rng.choice(np.arange(x_fit.shape[0]), size=probe_count, replace=False)
        x_probe = x_fit[probe_idx]
        y_probe = y_fit[probe_idx]
        p0 = model.param_vector()
        fdiag = estimate_fisher_info_with_diagnostics(lambda p: model.logpdf_vector(p, x_probe, y_probe), p0)
        fisher = fdiag.fisher_matrix
        fres = fisher_param_code_bits(
            fisher=fisher,
            n_samples=x_fit.shape[0],
            lambda_prior=1e-4,
            jitter=fisher_jitter,
        )
        dl_param = quantized_param_bits if strict_param_code else fres.dl_param_bits
        fisher_stress_bits = fres.dl_param_bits
        fisher_logdet = fres.logdet
        fisher_jit = fres.jitter_used
        fisher_eig_min = fres.eig_min
        fisher_eig_max = fres.eig_max
        fisher_cond_number = fres.cond_number
        fisher_effective_rank = float(fres.effective_rank)
        fisher_grad_norm_mean = fdiag.grad_norm_mean
        fisher_grad_norm_std = fdiag.grad_norm_std
        fisher_grad_norm_max = fdiag.grad_norm_max
    else:
        dl_param = (
            quantized_param_bits
            if strict_param_code
            else uniform_param_code_bits(k=model.n_params, n_samples=x_fit.shape[0])
        )
        fisher_stress_bits = np.nan
        fisher_logdet = np.nan
        fisher_jit = np.nan
        probe_count = 0
        fisher_eig_min = np.nan
        fisher_eig_max = np.nan
        fisher_cond_number = np.nan
        fisher_effective_rank = np.nan
        fisher_grad_norm_mean = np.nan
        fisher_grad_norm_std = np.nan
        fisher_grad_norm_max = np.nan

    breakdown = build_breakdown(dl_grammar, dl_struct, dl_param, dres)
    return {
        "candidate_id": spec.candidate_id,
        "model": model,
        "expr": expr,
        "dres": dres,
        "dl_grammar": dl_grammar,
        "dl_struct": dl_struct,
        "dl_param": dl_param,
        "dl_data": dres.dl_data_bits,
        "dl_total": breakdown.dl_total,
        "fisher_logdet": fisher_logdet,
        "fisher_jitter": fisher_jit,
        "fisher_eig_min": fisher_eig_min,
        "fisher_eig_max": fisher_eig_max,
        "fisher_cond_number": fisher_cond_number,
        "fisher_effective_rank": fisher_effective_rank,
        "fisher_grad_norm_mean": fisher_grad_norm_mean,
        "fisher_grad_norm_std": fisher_grad_norm_std,
        "fisher_grad_norm_max": fisher_grad_norm_max,
        "fisher_probe_seed": int(fisher_seed),
        "fisher_probe_count": int(probe_count),
        "fisher_stress_bits": float(fisher_stress_bits) if np.isfinite(fisher_stress_bits) else np.nan,
        "param_code_mode": "fixed_grid_elias_delta" if strict_param_code else "legacy_fisher_approximation",
        "rmse_eval": _rmse(y_eval, mu_eval),
        "nll_eval": _nll_student(y_eval, mu_eval, sigma_eval),
    }


def run_eidos_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    feature_cols: list[str],
    target_col: str,
    env_col: str,
    split_col: str,
    delta_y: float,
    seed: int,
    cfg: dict[str, Any],
    ledger: LedgerWriter,
    run_id: str,
    git_hash: str,
    ablation: EngineAblation | None = None,
    model_id: str = "eidos_full",
    var_units: dict[str, dict[str, float]] | None = None,
    target_unit: dict[str, float] | None = None,
    audit_trace: bool = False,
) -> EngineResult:
    ablation = ablation or EngineAblation()
    t0 = time.time()
    search_cfg = cfg["search"]
    inv_cfg = cfg["invariance"]
    conf_cfg = cfg["conformal"]
    device = "cpu"
    if cfg.get("resources", {}).get("prefer_gpu", True):
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

    observed_roles = set(df[split_col].astype(str).unique())
    strict_protocol = {"disc", "diag", "audit", "cal", "test"}.issubset(observed_roles)
    discovery_role = "disc" if strict_protocol else "work"
    audit_role = "audit" if strict_protocol else "lock"
    df_work = df[df[split_col] == discovery_role].copy()
    df_diag = df[df[split_col] == "diag"].copy() if strict_protocol else df_work
    df_lock = df[df[split_col] == audit_role].copy()
    df_cal = df[df[split_col] == "cal"].copy()
    df_test = df[df[split_col] == "test"].copy()
    unit_map = {c: dict(UNIT_DIMENSIONLESS) for c in feature_cols}
    if isinstance(var_units, dict):
        for c in feature_cols:
            raw = var_units.get(c)
            if isinstance(raw, dict):
                unit_map[c] = {str(k): float(v) for k, v in raw.items()}
    target_unit_map = dict(UNIT_DIMENSIONLESS)
    if isinstance(target_unit, dict):
        target_unit_map = {str(k): float(v) for k, v in target_unit.items()}

    candidates = default_candidate_library(feature_cols)
    if bool(search_cfg.get("enable_adaptive_candidates", False)):
        top_n = int(search_cfg.get("adaptive_top_features", 8))
        max_interactions = int(search_cfg.get("adaptive_max_interactions", 24))
        trig_scales = search_cfg.get("adaptive_trig_scales", [0.8, 1.0, 1.3, 1.8, 2.0, 2.4])
        if not isinstance(trig_scales, list):
            trig_scales = [0.8, 1.0, 1.3, 1.8, 2.0, 2.4]
        priority = _priority_features_by_correlation(
            df_work=df_work,
            feature_cols=feature_cols,
            target_col=target_col,
            top_n=top_n,
        )
        if bool(search_cfg.get("adaptive_drop_collinear", False)):
            priority = _prune_collinear_features(
                df_work=df_work,
                candidate_features=priority,
                threshold=float(search_cfg.get("adaptive_collinear_threshold", 0.98)),
            )
        candidates.extend(
            adaptive_trig_candidate_library(
                feature_cols=feature_cols,
                priority_features=priority,
                trig_scales=[float(v) for v in trig_scales],
                max_interactions=max_interactions,
                linear_ridge=float(search_cfg.get("adaptive_linear_ridge", 0.5)),
                trig_ridge=float(search_cfg.get("adaptive_trig_ridge", 1.0)),
                trig2_ridge=float(search_cfg.get("adaptive_trig2_ridge", 1.2)),
                poly_ridge=float(search_cfg.get("adaptive_poly_ridge", 2.0)),
                poly_hetero_ridge=float(search_cfg.get("adaptive_poly_hetero_ridge", 2.2)),
            )
        )
    if bool(search_cfg.get("enable_structural_boost", False)):
        top_n = int(search_cfg.get("structural_top_features", 6))
        quantiles = search_cfg.get("structural_center_quantiles", [0.2, 0.5, 0.8])
        if not isinstance(quantiles, list):
            quantiles = [0.2, 0.5, 0.8]
        priority = _priority_features_by_correlation(
            df_work=df_work,
            feature_cols=feature_cols,
            target_col=target_col,
            top_n=top_n,
        )
        if bool(search_cfg.get("structural_drop_collinear", False)):
            priority = _prune_collinear_features(
                df_work=df_work,
                candidate_features=priority,
                threshold=float(search_cfg.get("structural_collinear_threshold", 0.98)),
            )
        centers_map: dict[str, list[float]] = {}
        scales_map: dict[str, float] = {}
        for c in priority:
            x = df_work[c].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            q_vals: list[float] = []
            for q in quantiles:
                try:
                    qv = float(np.quantile(x, float(q)))
                except Exception:
                    continue
                if not np.isfinite(qv):
                    continue
                if any(abs(qv - v) < 1e-9 for v in q_vals):
                    continue
                q_vals.append(qv)
            if not q_vals:
                q_vals = [float(np.median(x))]
            centers_map[c] = q_vals
            q75 = float(np.quantile(x, 0.75))
            q25 = float(np.quantile(x, 0.25))
            iqr = max(1e-6, q75 - q25)
            std = float(np.std(x))
            scales_map[c] = float(max(1e-3, std, iqr / 1.349))
        dense_centers_map: dict[str, list[float]] = {}
        if bool(search_cfg.get("enable_structural_dense_rbf", False)):
            raw_dense_q = search_cfg.get(
                "structural_dense_rbf_quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            if not isinstance(raw_dense_q, list):
                raw_dense_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for c in priority:
                x = df_work[c].to_numpy(float)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                q_vals: list[float] = []
                for q in raw_dense_q:
                    try:
                        qv = float(np.quantile(x, float(q)))
                    except Exception:
                        continue
                    if not np.isfinite(qv):
                        continue
                    if any(abs(qv - v) < 1e-9 for v in q_vals):
                        continue
                    q_vals.append(qv)
                if not q_vals:
                    q_vals = [float(np.median(x))]
                dense_centers_map[c] = q_vals
        prototype_centers: list[dict[str, float]] = []
        if bool(search_cfg.get("enable_structural_proto_rbf", False)):
            proto_top = int(search_cfg.get("structural_proto_rbf_top_features", 8))
            proto_feats = priority[: min(max(1, proto_top), len(priority))]
            max_centers = max(1, int(search_cfg.get("structural_proto_rbf_centers", 64)))
            if proto_feats:
                x_proto = df_work[proto_feats].to_numpy(float)
                finite = np.isfinite(x_proto).all(axis=1)
                x_valid = x_proto[finite]
                if x_valid.shape[0] > 0:
                    med = np.median(x_valid, axis=0, keepdims=True)
                    start = int(np.argmin(np.sum((x_valid - med) ** 2, axis=1)))
                    chosen = [start]
                    min_d2 = np.sum((x_valid - x_valid[start : start + 1]) ** 2, axis=1)
                    target_centers = min(max_centers, x_valid.shape[0])
                    while len(chosen) < target_centers:
                        nxt = int(np.argmax(min_d2))
                        if nxt in chosen:
                            break
                        chosen.append(nxt)
                        min_d2 = np.minimum(min_d2, np.sum((x_valid - x_valid[nxt : nxt + 1]) ** 2, axis=1))
                    prototype_centers = [
                        {str(c): float(x_valid[idx, j]) for j, c in enumerate(proto_feats)} for idx in chosen
                    ]
        raw_scales = search_cfg.get("structural_rbf_scale_multipliers", [0.5, 1.0, 2.0])
        if not isinstance(raw_scales, list):
            raw_scales = [0.5, 1.0, 2.0]
        raw_dense_scales = search_cfg.get("structural_dense_rbf_scale_multipliers", [0.5, 1.0, 2.0])
        if not isinstance(raw_dense_scales, list):
            raw_dense_scales = [0.5, 1.0, 2.0]
        candidates.extend(
            structural_boost_candidate_library(
                feature_cols=feature_cols,
                priority_features=priority,
                feature_centers=centers_map,
                feature_scales=scales_map,
                dense_feature_centers=dense_centers_map,
                rbf_scale_multipliers=[float(v) for v in raw_scales],
                max_rbf_terms=int(search_cfg.get("structural_max_rbf_terms", 18)),
                max_hinge_terms=int(search_cfg.get("structural_max_hinge_terms", 12)),
                max_ratio_terms=int(search_cfg.get("structural_max_ratio_terms", 12)),
                max_pair_terms=int(search_cfg.get("structural_max_pair_terms", 6)),
                linear_ridge=float(search_cfg.get("structural_linear_ridge", 0.25)),
                poly3_ridge=float(search_cfg.get("structural_poly3_ridge", 0.6)),
                rbf_ridge=float(search_cfg.get("structural_rbf_ridge", 0.9)),
                rbf_hetero_ridge=float(search_cfg.get("structural_rbf_hetero_ridge", 1.2)),
                mix_ridge=float(search_cfg.get("structural_mix_ridge", 1.0)),
                enable_dense_rbf=bool(search_cfg.get("enable_structural_dense_rbf", False)),
                dense_rbf_top_features=int(search_cfg.get("structural_dense_rbf_top_features", 0)),
                dense_rbf_quantiles=[float(v) for v in search_cfg.get("structural_dense_rbf_quantiles", [])]
                if isinstance(search_cfg.get("structural_dense_rbf_quantiles", []), list)
                else None,
                dense_rbf_scale_multipliers=[float(v) for v in raw_dense_scales],
                dense_rbf_ridge=float(search_cfg.get("structural_dense_rbf_ridge", 0.2)),
                enable_pairwise_dense_rbf=bool(search_cfg.get("enable_structural_pairwise_dense_rbf", False)),
                dense_pair_rbf_top_features=int(search_cfg.get("structural_pairwise_dense_rbf_top_features", 4)),
                dense_pair_rbf_max_terms=int(search_cfg.get("structural_pairwise_dense_rbf_max_terms", 72)),
                dense_pair_rbf_ridge=float(search_cfg.get("structural_pairwise_dense_rbf_ridge", 0.12)),
                enable_step_grid=bool(search_cfg.get("enable_structural_step_grid", False)),
                step_grid_top_features=int(search_cfg.get("structural_step_grid_top_features", 6)),
                step_grid_max_terms=int(search_cfg.get("structural_step_grid_max_terms", 48)),
                step_pair_max_terms=int(search_cfg.get("structural_step_pair_max_terms", 48)),
                step_grid_ridge=float(search_cfg.get("structural_step_grid_ridge", 0.18)),
                enable_dense_rbf_local_linear=bool(search_cfg.get("enable_structural_dense_rbf_local_linear", False)),
                dense_local_linear_max_terms=int(search_cfg.get("structural_dense_rbf_local_linear_max_terms", 96)),
                dense_local_linear_ridge=float(search_cfg.get("structural_dense_rbf_local_linear_ridge", 0.06)),
                enable_fourier_mix=bool(search_cfg.get("enable_structural_fourier_mix", False)),
                fourier_top_features=int(search_cfg.get("structural_fourier_top_features", 8)),
                fourier_max_terms=int(search_cfg.get("structural_fourier_max_terms", 72)),
                fourier_ridge=float(search_cfg.get("structural_fourier_ridge", 0.08)),
                prototype_centers=prototype_centers,
                enable_proto_rbf=bool(search_cfg.get("enable_structural_proto_rbf", False)),
                proto_rbf_scale_multipliers=[float(v) for v in search_cfg.get("structural_proto_rbf_scale_multipliers", [0.7, 1.2, 2.0])]
                if isinstance(search_cfg.get("structural_proto_rbf_scale_multipliers", []), list)
                else None,
                proto_rbf_max_terms=int(search_cfg.get("structural_proto_rbf_max_terms", 160)),
                proto_rbf_ridge=float(search_cfg.get("structural_proto_rbf_ridge", 0.04)),
                enable_proto_rbf_local_linear=bool(search_cfg.get("enable_structural_proto_rbf_local_linear", False)),
                proto_rbf_local_linear_max_terms=int(
                    search_cfg.get("structural_proto_rbf_local_linear_max_terms", 120)
                ),
                proto_rbf_local_linear_ridge=float(search_cfg.get("structural_proto_rbf_local_linear_ridge", 0.05)),
                enable_competitive_union=bool(search_cfg.get("enable_structural_competitive_union", False)),
                competitive_union_max_terms=int(search_cfg.get("structural_competitive_union_max_terms", 360)),
                competitive_union_ridge=float(search_cfg.get("structural_competitive_union_ridge", 0.08)),
            )
        )
    if bool(search_cfg.get("enable_partition_candidates", False)):
        candidates.extend(
            _partition_candidate_library(
                df_work=df_work,
                feature_cols=feature_cols,
                target_col=target_col,
                search_cfg=search_cfg,
                seed=seed,
            )
        )
    if bool(search_cfg.get("enable_physics_templates", False)):
        top_n = int(search_cfg.get("physics_top_features", 4))
        priority = _priority_features_by_correlation(
            df_work=df_work,
            feature_cols=feature_cols,
            target_col=target_col,
            top_n=top_n,
        )
        candidates.extend(
            physics_template_candidate_library(
                feature_cols=feature_cols,
                priority_features=priority,
                top_features=top_n,
                inv_ridge=float(search_cfg.get("physics_inv_ridge", 0.3)),
                rel_ridge=float(search_cfg.get("physics_rel_ridge", 0.45)),
                osc_ridge=float(search_cfg.get("physics_osc_ridge", 0.35)),
                planck_ridge=float(search_cfg.get("physics_planck_ridge", 0.5)),
                pair_ridge=float(search_cfg.get("physics_pair_ridge", 0.4)),
            )
        )
    forced_candidate_id = search_cfg.get("forced_candidate_id")
    if not bool(search_cfg.get("allow_forced_candidate", False)):
        forced_candidate_id = None
    if forced_candidate_id is not None:
        forced_candidate_id = str(forced_candidate_id)

    always_include_raw = search_cfg.get("always_include_candidate_ids", [])
    if isinstance(always_include_raw, str):
        always_include_before_screen = {v.strip() for v in always_include_raw.split(",") if v.strip()}
    elif isinstance(always_include_raw, list):
        always_include_before_screen = {str(v).strip() for v in always_include_raw if str(v).strip()}
    else:
        always_include_before_screen = set()
    if forced_candidate_id:
        always_include_before_screen.add(str(forced_candidate_id))
    max_cands = int(search_cfg.get("max_candidates", len(candidates)))
    if len(candidates) > max_cands:
        head = candidates[:max_cands]
        head_ids = {c.candidate_id for c in head}
        extras = [
            c
            for c in candidates[max_cands:]
            if c.candidate_id in always_include_before_screen and c.candidate_id not in head_ids
        ]
        candidates = head + extras
    else:
        candidates = candidates[:max_cands]

    screen = _screen_candidates(
        candidates=candidates,
        df_work=df_work,
        feature_cols=feature_cols,
        clip_ratio_threshold=float(search_cfg.get("clip_ratio_threshold", 0.15)),
        seed=seed,
        forced_candidate_id=forced_candidate_id,
        var_units=unit_map,
    )
    candidates = screen.accepted
    if not candidates:
        raise RuntimeError("All candidates rejected by DSL/type/unit/domain screening.")
    for rej in screen.rejected:
        ledger.log(
            {
                "run_id": run_id,
                "git_hash": git_hash,
                "seed": seed,
                "dataset_id": dataset_id,
                "env_id": "all",
                "split": "work",
                "delta_y_source": "metadata",
                "delta_y_value": delta_y,
                "model_id": model_id,
                "candidate_id": rej["candidate_id"],
                "dl_struct": None,
                "dl_param": None,
                "dl_data": None,
                "dl_total": None,
                "inv_test_type": "screening",
                "p_inv_work": None,
                "p_inv_lock": None,
                "conformal_coverage": None,
                "conformal_length": None,
                "failure_reason": rej["failure_reason"],
                "notes": f"candidate_rejected clip_ratio_max={rej['clip_ratio_max']:.6f}",
            }
        )

    # Layer 1: equal-size env subsampling + optional unbiased scaling
    layer1_n = int(search_cfg.get("layer1_subsample_per_env", 96))
    sub = _stratified_equal_subsample(df_work, env_col=env_col, n_per_env=layer1_n, seed=seed)
    l1_scores = []
    l1_start = time.time()
    for spec in candidates:
        out = _fit_score_single(
            spec=spec,
            df_fit=sub,
            df_eval=sub,
            feature_cols=feature_cols,
            target_col=target_col,
            delta_y=delta_y,
            use_bin_likelihood=ablation.use_bin_likelihood,
            use_fisher=False,
            fisher_probe_points=64,
            fisher_jitter=float(search_cfg.get("fisher_jitter", 1e-4)),
            robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
            device=device,
            max_fisher_params=int(search_cfg.get("max_fisher_params", 28)),
            fisher_seed=_stable_seed(seed, spec.candidate_id, salt=11),
            strict_param_code=strict_protocol,
            fixed_param_step=float(search_cfg.get("fixed_param_step", 1e-3)),
        )
        model = out["model"]
        mu_sub, sigma_sub = model.predict_mu_sigma(sub[feature_cols].to_numpy(float))
        if ablation.use_bin_likelihood:
            bits_sub = -np.log2(
                np.clip(
                    np.exp2(-dl_data_bin_likelihood(
                        sub[target_col].to_numpy(float), mu_sub, sigma_sub, delta_y, robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4))
                    ).avg_bits),
                    1e-12,
                    1.0,
                )
            )
            # stable per-sample recomputation
            from experiments.interval_core.mdl.data_code import bin_prob_mass

            probs = bin_prob_mass(
                y_tilde=sub[target_col].to_numpy(float),
                mu=mu_sub,
                sigma=sigma_sub,
                delta_y=delta_y,
                robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
            )
            bits_sub = -np.log2(probs)
        else:
            from scipy.stats import t

            z = (sub[target_col].to_numpy(float) - mu_sub) / np.maximum(sigma_sub, 1e-8)
            bits_sub = -np.log2(np.clip(t.pdf(z, df=3.0) / np.maximum(sigma_sub, 1e-8), 1e-300, 1e300))

        scaled_dl_data = _scaled_sub_dl(
            full_work=df_work,
            sub_work=sub,
            bits_sub_per_sample=bits_sub,
            env_col=env_col,
            use_unbiased_scaling=ablation.use_unbiased_scaling,
        )
        l1_scores.append(
            {
                "candidate_id": spec.candidate_id,
                "score_l1": scaled_dl_data + out["dl_struct"] + out["dl_param"],
            }
        )
    layer1_time = time.time() - l1_start

    l1_df = pd.DataFrame(l1_scores).sort_values("score_l1")
    n_layer2 = max(int(search_cfg.get("top_k", 12)) * 3, 10)
    layer1_exclude_raw = search_cfg.get("layer1_exclude_candidate_ids", [])
    if isinstance(layer1_exclude_raw, str):
        layer1_exclude_ids = {v.strip() for v in layer1_exclude_raw.split(",") if v.strip()}
    elif isinstance(layer1_exclude_raw, list):
        layer1_exclude_ids = {str(v).strip() for v in layer1_exclude_raw if str(v).strip()}
    else:
        layer1_exclude_ids = set()
    l1_rank_df = l1_df[~l1_df["candidate_id"].isin(layer1_exclude_ids)].copy()
    keep_ids = set(l1_rank_df.head(n_layer2)["candidate_id"].tolist())
    always_include_raw = search_cfg.get("always_include_candidate_ids", [])
    if isinstance(always_include_raw, str):
        always_include_ids = [v.strip() for v in always_include_raw.split(",") if v.strip()]
    elif isinstance(always_include_raw, list):
        always_include_ids = [str(v).strip() for v in always_include_raw if str(v).strip()]
    else:
        always_include_ids = []
    keep_ids.update(always_include_ids)
    if forced_candidate_id:
        keep_ids.add(str(forced_candidate_id))
    layer2_specs = [c for c in candidates if c.candidate_id in keep_ids]
    # Hard safety gate: EIDOS search must stay inside native symbolic candidates.
    if any(str(s.candidate_id).startswith("meta_") for s in layer2_specs):
        raise RuntimeError("Invalid EIDOS candidate set: meta_* candidates are not allowed in strict mode.")

    # Layer 2: full work evaluation with invariance and full MDL
    layer2_rows: list[dict[str, Any]] = []
    lock_cert_cache: dict[str, float] = {}
    lock_stat_cache: dict[str, float] = {}
    conf_cert_cache: dict[str, tuple[float, float]] = {}
    layer2_start = time.time()
    selection_policy = str(search_cfg.get("selection_policy", "dl_only"))
    for spec in layer2_specs:
        out = _fit_score_single(
            spec=spec,
            df_fit=df_work,
            df_eval=df_diag,
            feature_cols=feature_cols,
            target_col=target_col,
            delta_y=delta_y,
            use_bin_likelihood=ablation.use_bin_likelihood,
            use_fisher=ablation.use_fisher,
            fisher_probe_points=int(search_cfg.get("fisher_probe_points", 256)),
            fisher_jitter=float(search_cfg.get("fisher_jitter", 1e-4)),
            robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
            device=device,
            max_fisher_params=int(search_cfg.get("max_fisher_params", 28)),
            fisher_seed=_stable_seed(seed, spec.candidate_id, salt=23),
            strict_param_code=strict_protocol,
            fixed_param_step=float(search_cfg.get("fixed_param_step", 1e-3)),
        )

        if strict_protocol:
            rmse_crossfit_work = float("nan")
            if ablation.use_crossfit and selection_policy.startswith("cv_"):
                cf = crossfit_residuals(
                    df=df_work,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    env_col=env_col,
                    model_factory=lambda s=spec: LinearBasisModel(s, feature_cols, ridge=s.ridge, device=device),
                    k=int(search_cfg.get("crossfit_k", 2)),
                    seed=seed,
                )
                rmse_crossfit_work = _rmse(df_work[target_col].to_numpy(float), cf.mu)
            model = out["model"]
            mu_diag, sigma_diag = model.predict_mu_sigma(df_diag[feature_cols].to_numpy(float))
            z = (df_diag[target_col].to_numpy(float) - mu_diag) / np.maximum(sigma_diag, 1e-8)
            z = normalize_residuals_by_env(
                z,
                df_diag[env_col].to_numpy(int),
                mode=str(inv_cfg.get("residual_normalization", "global")),
                min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
            )
            inv = mmd_rff_permutation_test(
                z=z,
                env=df_diag[env_col].to_numpy(int),
                n_features=int(inv_cfg.get("rff_features", 128)),
                n_perm=int(inv_cfg.get("work_permutations", 100)),
                seed=seed,
            )
            p_inv_work = float(inv.p_value)
        elif ablation.use_crossfit:
            cf = crossfit_residuals(
                df=df_work,
                feature_cols=feature_cols,
                target_col=target_col,
                env_col=env_col,
                model_factory=lambda s=spec: LinearBasisModel(s, feature_cols, ridge=s.ridge, device=device),
                # keep same hardware path for per-fold fitting
                k=int(search_cfg.get("crossfit_k", 2)),
                seed=seed,
            )
            rmse_crossfit_work = _rmse(df_work[target_col].to_numpy(float), cf.mu)
            z_cf = normalize_residuals_by_env(
                cf.z,
                cf.env,
                mode=str(inv_cfg.get("residual_normalization", "global")),
                min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
            )
            inv = mmd_rff_permutation_test(
                z=z_cf,
                env=cf.env,
                n_features=int(inv_cfg.get("rff_features", 128)),
                n_perm=int(inv_cfg.get("work_permutations", 100)),
                seed=seed,
                permute_blocks=cf.fold,
            )
            p_inv_work = float(inv.p_value)
        else:
            rmse_crossfit_work = float("nan")
            model = out["model"]
            mu_w, sigma_w = model.predict_mu_sigma(df_work[feature_cols].to_numpy(float))
            z = (df_work[target_col].to_numpy(float) - mu_w) / np.maximum(sigma_w, 1e-8)
            z = normalize_residuals_by_env(
                z,
                df_work[env_col].to_numpy(int),
                mode=str(inv_cfg.get("residual_normalization", "global")),
                min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
            )
            inv = mmd_rff_permutation_test(
                z=z,
                env=df_work[env_col].to_numpy(int),
                n_features=int(inv_cfg.get("rff_features", 128)),
                n_perm=int(inv_cfg.get("work_permutations", 100)),
                seed=seed,
            )
            p_inv_work = float(inv.p_value)

        row = {
            "candidate_id": spec.candidate_id,
            "model": out["model"],
            "dl_total": out["dl_total"],
            "dl_grammar": out["dl_grammar"],
            "dl_struct": out["dl_struct"],
            "dl_param": out["dl_param"],
            "dl_data": out["dl_data"],
            "p_inv_work": p_inv_work,
            "inv_stat_work": float(inv.stat),
            "fisher_logdet": out["fisher_logdet"],
            "fisher_jitter": out["fisher_jitter"],
            "fisher_eig_min": out["fisher_eig_min"],
            "fisher_eig_max": out["fisher_eig_max"],
            "fisher_cond_number": out["fisher_cond_number"],
            "fisher_effective_rank": out["fisher_effective_rank"],
            "fisher_grad_norm_mean": out["fisher_grad_norm_mean"],
            "fisher_grad_norm_std": out["fisher_grad_norm_std"],
            "fisher_grad_norm_max": out["fisher_grad_norm_max"],
            "fisher_probe_seed": out["fisher_probe_seed"],
            "fisher_probe_count": out["fisher_probe_count"],
            "rmse_work": out["rmse_eval"],
            "rmse_crossfit_work": rmse_crossfit_work,
            "nll_work": out["nll_eval"],
        }
        layer2_rows.append(row)
        ledger.log(
            {
                "run_id": run_id,
                "git_hash": git_hash,
                "seed": seed,
                "dataset_id": dataset_id,
                "env_id": "all",
                "split": "diag" if strict_protocol else "work",
                "delta_y_source": "metadata",
                "delta_y_value": delta_y,
                "model_id": model_id,
                "candidate_id": spec.candidate_id,
                "dl_grammar": float(out["dl_grammar"]),
                "dl_struct": float(out["dl_struct"]),
                "dl_param": float(out["dl_param"]),
                "dl_data": float(out["dl_data"]),
                "dl_total": float(out["dl_total"]),
                "inv_test_type": "mmd_rff",
                "p_inv_work": float(p_inv_work),
                "inv_stat_work": float(inv.stat),
                "p_inv_lock": None,
                "conformal_coverage": None,
                "conformal_length": None,
                "failure_reason": None,
                "notes": "layer2_candidate",
                "fisher_logdet": float(out["fisher_logdet"]) if np.isfinite(out["fisher_logdet"]) else None,
                "fisher_jitter": float(out["fisher_jitter"]) if np.isfinite(out["fisher_jitter"]) else None,
                "fisher_eig_min": float(out["fisher_eig_min"]) if np.isfinite(out["fisher_eig_min"]) else None,
                "fisher_eig_max": float(out["fisher_eig_max"]) if np.isfinite(out["fisher_eig_max"]) else None,
                "fisher_cond_number": float(out["fisher_cond_number"])
                if np.isfinite(out["fisher_cond_number"])
                else None,
                "fisher_effective_rank": float(out["fisher_effective_rank"])
                if np.isfinite(out["fisher_effective_rank"])
                else None,
                "fisher_grad_norm_mean": float(out["fisher_grad_norm_mean"])
                if np.isfinite(out["fisher_grad_norm_mean"])
                else None,
                "fisher_probe_seed": int(out["fisher_probe_seed"]),
                "fisher_probe_count": int(out["fisher_probe_count"]),
                "rmse_crossfit_work": float(rmse_crossfit_work) if np.isfinite(rmse_crossfit_work) else None,
            }
        )

    # GrammarInducer outer loop (single-step): mine reusable concepts from top candidates,
    # expand candidate set, and keep only if full-work DL improves by epsilon.
    # The strict protocol keeps the grammar fixed.  The legacy grammar-growth
    # loop queried audit/calibration/test certificates during search and is
    # therefore disabled whenever five evidence roles are present.
    use_grammar_inducer = bool(search_cfg.get("enable_grammar_inducer", True)) and not strict_protocol
    if use_grammar_inducer and layer2_specs:
        max_new_concepts = int(search_cfg.get("max_new_concepts", 2))
        min_reuse = int(search_cfg.get("concept_min_reuse", 2))
        max_augmented = int(search_cfg.get("concept_augmented_candidates", 12))
        eps_dl = float(search_cfg.get("concept_eps_dl", 1.0))
        concepts = _mine_reusable_concepts(
            specs=layer2_specs,
            max_new_concepts=max_new_concepts,
            min_reuse=min_reuse,
            var_units=unit_map,
        )
        if concepts:
            var_types = {c: "Real" for c in feature_cols}
            concept_var_units = {c: dict(unit_map.get(c, UNIT_DIMENSIONLESS)) for c in feature_cols}
            concept_expr_tokens = [canonicalize_syn(c).to_string() for c in concepts]
            type_tokens: set[str] = set()
            unit_tokens: set[str] = set()
            for cexpr in concepts:
                tu = infer_type_unit(cexpr, var_types=var_types, var_units=concept_var_units)
                if tu.valid:
                    type_tokens.add(str(tu.kind))
                    unit_tokens.add(_format_unit_signature(dict(tu.unit)))
                else:
                    type_tokens.add(f"invalid:{tu.reason or 'unknown'}")
                    unit_tokens.add("invalid")
                concept_expr_sig = ";".join(concept_expr_tokens)
                type_signature = "|".join(sorted(type_tokens)) if type_tokens else "Real"
                unit_signature = "|".join(sorted(unit_tokens)) if unit_tokens else "dimensionless"
                has_non_dimensionless_concept = any(
                    (u not in {"dimensionless", "invalid"}) for u in unit_tokens
                )

            aug_specs = _build_concept_augmented_specs(
                base_specs=layer2_specs,
                concepts=concepts,
                max_augmented=max_augmented,
            )
            aug_rows: list[dict[str, Any]] = []
            for spec in aug_specs:
                out = _fit_score_single(
                    spec=spec,
                    df_fit=df_work,
                    df_eval=df_work,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    delta_y=delta_y,
                    use_bin_likelihood=ablation.use_bin_likelihood,
                    use_fisher=ablation.use_fisher,
                    fisher_probe_points=int(search_cfg.get("fisher_probe_points", 256)),
                    fisher_jitter=float(search_cfg.get("fisher_jitter", 1e-4)),
                    robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
                    device=device,
                    max_fisher_params=int(search_cfg.get("max_fisher_params", 28)),
                    fisher_seed=_stable_seed(seed, spec.candidate_id, salt=37),
                    new_concepts=len(concepts),
                    strict_param_code=strict_protocol,
                    fixed_param_step=float(search_cfg.get("fixed_param_step", 1e-3)),
                )
                if ablation.use_crossfit:
                    cf = crossfit_residuals(
                        df=df_work,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        env_col=env_col,
                        model_factory=lambda s=spec: LinearBasisModel(s, feature_cols, ridge=s.ridge, device=device),
                        k=int(search_cfg.get("crossfit_k", 2)),
                        seed=seed,
                    )
                    z_cf = normalize_residuals_by_env(
                        cf.z,
                        cf.env,
                        mode=str(inv_cfg.get("residual_normalization", "global")),
                        min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                        shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
                    )
                    inv = mmd_rff_permutation_test(
                        z=z_cf,
                        env=cf.env,
                        n_features=int(inv_cfg.get("rff_features", 128)),
                        n_perm=int(inv_cfg.get("work_permutations", 100)),
                        seed=seed,
                        permute_blocks=cf.fold,
                    )
                    p_inv_work = float(inv.p_value)
                else:
                    model = out["model"]
                    mu_w, sigma_w = model.predict_mu_sigma(df_work[feature_cols].to_numpy(float))
                    z = (df_work[target_col].to_numpy(float) - mu_w) / np.maximum(sigma_w, 1e-8)
                    z = normalize_residuals_by_env(
                        z,
                        df_work[env_col].to_numpy(int),
                        mode=str(inv_cfg.get("residual_normalization", "global")),
                        min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                        shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
                    )
                    inv = mmd_rff_permutation_test(
                        z=z,
                        env=df_work[env_col].to_numpy(int),
                        n_features=int(inv_cfg.get("rff_features", 128)),
                        n_perm=int(inv_cfg.get("work_permutations", 100)),
                        seed=seed,
                    )
                    p_inv_work = float(inv.p_value)

                aug_rows.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "parent_candidate_id": spec.parent_candidate_id,
                        "concept_expr_canonical": concept_expr_sig,
                        "type_signature": type_signature,
                        "unit_signature": unit_signature,
                        "model": out["model"],
                        "dl_total": out["dl_total"],
                        "dl_grammar": out["dl_grammar"],
                        "dl_struct": out["dl_struct"],
                        "dl_param": out["dl_param"],
                        "dl_data": out["dl_data"],
                        "p_inv_work": p_inv_work,
                        "fisher_logdet": out["fisher_logdet"],
                        "fisher_jitter": out["fisher_jitter"],
                        "fisher_eig_min": out["fisher_eig_min"],
                        "fisher_eig_max": out["fisher_eig_max"],
                        "fisher_cond_number": out["fisher_cond_number"],
                        "fisher_effective_rank": out["fisher_effective_rank"],
                        "fisher_grad_norm_mean": out["fisher_grad_norm_mean"],
                        "fisher_grad_norm_std": out["fisher_grad_norm_std"],
                        "fisher_grad_norm_max": out["fisher_grad_norm_max"],
                        "fisher_probe_seed": out["fisher_probe_seed"],
                        "fisher_probe_count": out["fisher_probe_count"],
                        "rmse_work": out["rmse_eval"],
                        "nll_work": out["nll_eval"],
                    }
                )

            if aug_rows and layer2_rows:
                base_layer2_rows = list(layer2_rows)
                alpha_inv = float(inv_cfg.get("alpha_inv", 0.05))
                alpha_conf = _primary_conformal_alpha(conf_cfg)
                cov_tol = float(search_cfg.get("concept_conformal_cov_tol", 0.0))
                len_rel_tol = float(search_cfg.get("concept_conformal_len_rel_tol", 0.0))
                require_lock = bool(search_cfg.get("concept_require_lock_cert", True))
                require_conformal = bool(search_cfg.get("concept_require_conformal_cert", True))

                old_valid = [r for r in layer2_rows if float(r["p_inv_work"]) >= alpha_inv]
                new_valid = [r for r in aug_rows if float(r["p_inv_work"]) >= alpha_inv]
                old_best_row = min(old_valid or layer2_rows, key=lambda r: float(r["dl_total"]))
                new_best_row = min(new_valid or aug_rows, key=lambda r: float(r["dl_total"]))
                old_best = float(old_best_row["dl_total"])
                new_best = float(new_best_row["dl_total"])
                bank_lock_cache: dict[str, float] = {}
                bank_conf_cache: dict[str, tuple[float, float]] = {}

                def _get_certificate(row: dict[str, Any], cert_seed_salt: int) -> CandidateCertificate:
                    cid = str(row["candidate_id"])
                    model = row["model"]
                    if cid not in lock_cert_cache:
                        lock_res = _evaluate_lock_invariance_result(
                            model=model,
                            df_lock=df_lock,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            env_col=env_col,
                            inv_cfg=inv_cfg,
                            seed=_stable_seed(seed, cid, salt=cert_seed_salt),
                        )
                        lock_cert_cache[cid] = float(lock_res.p_value)
                        lock_stat_cache[cid] = float(lock_res.stat)
                    if cid not in conf_cert_cache:
                        conf_cert_cache[cid] = _evaluate_conformal_certificate(
                            model=model,
                            df_cal=df_cal,
                            df_test=df_test,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            alpha=alpha_conf,
                        )
                    cov, avg_len = conf_cert_cache[cid]
                    return CandidateCertificate(
                        p_inv_lock=float(lock_cert_cache[cid]),
                        conformal_coverage=float(cov),
                        conformal_length=float(avg_len),
                    )

                def _get_bank_certificate(row: dict[str, Any], cert_seed_salt: int) -> CandidateCertificate:
                    cid = str(row["candidate_id"])
                    model = row["model"]
                    if cid not in bank_lock_cache:
                        lock_res = _evaluate_lock_invariance_result(
                            model=model,
                            df_lock=df_lock,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            env_col=env_col,
                            inv_cfg=inv_cfg,
                            seed=_stable_seed(seed, cid, salt=cert_seed_salt),
                        )
                        bank_lock_cache[cid] = float(lock_res.p_value)
                    if cid not in bank_conf_cache:
                        bank_conf_cache[cid] = _evaluate_conformal_certificate(
                            model=model,
                            df_cal=df_cal,
                            df_test=df_test,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            alpha=alpha_conf,
                        )
                    cov, avg_len = bank_conf_cache[cid]
                    return CandidateCertificate(
                        p_inv_lock=float(bank_lock_cache[cid]),
                        conformal_coverage=float(cov),
                        conformal_length=float(avg_len),
                    )

                old_cert = _get_certificate(old_best_row, cert_seed_salt=101)
                new_cert = _get_certificate(new_best_row, cert_seed_salt=103)

                bonus_bits = 0.0
                if has_non_dimensionless_concept:
                    bonus_bits = float(search_cfg.get("concept_unit_bonus_bits", 0.0))
                parent_row_by_id = {str(r["candidate_id"]): r for r in base_layer2_rows}
                for row in aug_rows:
                    parent_id = str(row.get("parent_candidate_id") or "")
                    parent_row = parent_row_by_id.get(parent_id)
                    if parent_row is None:
                        bank_cert = CandidateCertificate(float("nan"), float("nan"), float("nan"))
                        parent_cert = CandidateCertificate(float("nan"), float("nan"), float("nan"))
                        delta_parent = float("nan")
                        bank_state = {
                            "passed": False,
                            "failure_reason": "missing_parent_candidate",
                        }
                    else:
                        bank_cert = _get_bank_certificate(row, cert_seed_salt=151)
                        parent_cert = _get_bank_certificate(parent_row, cert_seed_salt=157)
                        delta_parent = float(parent_row["dl_total"]) - float(row["dl_total"])
                        bank_state = _concept_bank_gate_status(
                            delta_dl_parent_bits=delta_parent,
                            candidate_cert=bank_cert,
                            parent_cert=parent_cert,
                            alpha_inv=alpha_inv,
                            eps_dl=eps_dl,
                            bonus_bits=bonus_bits,
                            cov_tol=cov_tol,
                            len_rel_tol=len_rel_tol,
                        )
                    ledger.log(
                        {
                            "run_id": run_id,
                            "git_hash": git_hash,
                            "seed": seed,
                            "dataset_id": dataset_id,
                            "env_id": "all",
                            "split": "work",
                            "delta_y_source": "metadata",
                            "delta_y_value": delta_y,
                            "model_id": model_id,
                            "candidate_id": row["candidate_id"],
                            "dl_grammar": float(row.get("dl_grammar", 0.0)),
                            "dl_struct": float(row["dl_struct"]),
                            "dl_param": float(row["dl_param"]),
                            "dl_data": float(row["dl_data"]),
                            "dl_total": float(row["dl_total"]),
                            "inv_test_type": "mmd_rff",
                            "p_inv_work": float(row["p_inv_work"]),
                            "p_inv_lock": float(bank_cert.p_inv_lock) if np.isfinite(bank_cert.p_inv_lock) else None,
                            "conformal_coverage": float(bank_cert.conformal_coverage)
                            if np.isfinite(bank_cert.conformal_coverage)
                            else None,
                            "conformal_length": float(bank_cert.conformal_length)
                            if np.isfinite(bank_cert.conformal_length)
                            else None,
                            "failure_reason": bank_state["failure_reason"],
                            "notes": "concept_candidate_eval",
                            "concept_expr": str(row.get("concept_expr_canonical", "")),
                            "concept_expr_canonical": str(row.get("concept_expr_canonical", "")),
                            "type_signature": str(row.get("type_signature", "")),
                            "unit_signature": str(row.get("unit_signature", "")),
                            "parent_candidate_id": parent_id or None,
                            "delta_dl_parent_bits": float(delta_parent) if np.isfinite(delta_parent) else None,
                            "parent_dl_total": float(parent_row["dl_total"]) if parent_row is not None else None,
                            "candidate_p_inv_lock": float(bank_cert.p_inv_lock) if np.isfinite(bank_cert.p_inv_lock) else None,
                            "parent_p_inv_lock": float(parent_cert.p_inv_lock) if np.isfinite(parent_cert.p_inv_lock) else None,
                            "candidate_conformal_coverage": float(bank_cert.conformal_coverage)
                            if np.isfinite(bank_cert.conformal_coverage)
                            else None,
                            "parent_conformal_coverage": float(parent_cert.conformal_coverage)
                            if np.isfinite(parent_cert.conformal_coverage)
                            else None,
                            "candidate_conformal_length": float(bank_cert.conformal_length)
                            if np.isfinite(bank_cert.conformal_length)
                            else None,
                            "parent_conformal_length": float(parent_cert.conformal_length)
                            if np.isfinite(parent_cert.conformal_length)
                            else None,
                            "bank_gate_pass": bool(bank_state["passed"]),
                            "fisher_logdet": float(row["fisher_logdet"]) if np.isfinite(row["fisher_logdet"]) else None,
                            "fisher_jitter": float(row["fisher_jitter"]) if np.isfinite(row["fisher_jitter"]) else None,
                            "fisher_eig_min": float(row["fisher_eig_min"]) if np.isfinite(row["fisher_eig_min"]) else None,
                            "fisher_eig_max": float(row["fisher_eig_max"]) if np.isfinite(row["fisher_eig_max"]) else None,
                            "fisher_cond_number": float(row["fisher_cond_number"])
                            if np.isfinite(row["fisher_cond_number"])
                            else None,
                            "fisher_effective_rank": float(row["fisher_effective_rank"])
                            if np.isfinite(row["fisher_effective_rank"])
                            else None,
                            "fisher_grad_norm_mean": float(row["fisher_grad_norm_mean"])
                            if np.isfinite(row["fisher_grad_norm_mean"])
                            else None,
                            "fisher_probe_seed": int(row["fisher_probe_seed"]),
                            "fisher_probe_count": int(row["fisher_probe_count"]),
                        }
                    )
                concept_dl_ok = new_best <= old_best - max(0.0, eps_dl - bonus_bits)
                concept_lock_ok = (not require_lock) or (
                    np.isfinite(new_cert.p_inv_lock) and (new_cert.p_inv_lock >= alpha_inv)
                )
                concept_conf_cov_ok = (not require_conformal) or (
                    np.isfinite(new_cert.conformal_coverage)
                    and np.isfinite(old_cert.conformal_coverage)
                    and (new_cert.conformal_coverage >= old_cert.conformal_coverage - cov_tol)
                )
                concept_conf_len_ok = (not require_conformal) or (
                    np.isfinite(new_cert.conformal_length)
                    and np.isfinite(old_cert.conformal_length)
                    and (new_cert.conformal_length <= old_cert.conformal_length * (1.0 + len_rel_tol))
                )

                concept_commit = concept_dl_ok and concept_lock_ok and concept_conf_cov_ok and concept_conf_len_ok
                if concept_commit:
                    layer2_rows.extend(aug_rows)
                fail_reasons: list[str] = []
                if not concept_dl_ok:
                    fail_reasons.append("dl_not_improved")
                if not concept_lock_ok:
                    fail_reasons.append("lock_invariance_fail")
                if not concept_conf_cov_ok:
                    fail_reasons.append("conformal_coverage_drop")
                if not concept_conf_len_ok:
                    fail_reasons.append("conformal_length_increase")
                failure_reason = None if concept_commit else "|".join(fail_reasons) or "concept_rollback"
                decision = "commit" if concept_commit else "rollback"
                ledger.log(
                    {
                        "run_id": run_id,
                        "git_hash": git_hash,
                        "seed": seed,
                        "dataset_id": dataset_id,
                        "env_id": "all",
                        "split": "work",
                        "delta_y_source": "metadata",
                        "delta_y_value": delta_y,
                        "model_id": model_id,
                        "candidate_id": f"concept_{decision}:{len(concepts)}",
                        "dl_grammar": float(new_best_row.get("dl_grammar", 0.0)),
                        "dl_struct": None,
                        "dl_param": None,
                        "dl_data": None,
                        "dl_total": float(new_best),
                        "inv_test_type": "concept_growth",
                        "p_inv_work": float(new_best_row["p_inv_work"]),
                        "p_inv_lock": float(new_cert.p_inv_lock) if np.isfinite(new_cert.p_inv_lock) else None,
                        "conformal_coverage": float(new_cert.conformal_coverage)
                        if np.isfinite(new_cert.conformal_coverage)
                        else None,
                        "conformal_length": float(new_cert.conformal_length)
                        if np.isfinite(new_cert.conformal_length)
                        else None,
                        "failure_reason": failure_reason,
                        "concept_expr": concept_expr_sig,
                        "type_signature": type_signature,
                        "unit_signature": unit_signature,
                        "notes": (
                            f"concept_growth_{decision} n_concepts={len(concepts)} "
                            f"old_best_id={old_best_row['candidate_id']} new_best_id={new_best_row['candidate_id']} "
                            f"old_best_dl={old_best:.6f} new_best_dl={new_best:.6f} eps_dl={eps_dl:.6f} bonus_bits={bonus_bits:.6f} "
                            f"old_p_lock={old_cert.p_inv_lock:.6f} new_p_lock={new_cert.p_inv_lock:.6f} alpha_inv={alpha_inv:.6f} "
                            f"old_cov={old_cert.conformal_coverage:.6f} new_cov={new_cert.conformal_coverage:.6f} cov_tol={cov_tol:.6f} "
                            f"old_len={old_cert.conformal_length:.6f} new_len={new_cert.conformal_length:.6f} len_rel_tol={len_rel_tol:.6f}"
                        ),
                    }
                )

    layer2_time = time.time() - layer2_start
    l2 = pd.DataFrame(layer2_rows).sort_values("dl_total")
    alpha_inv = float(inv_cfg.get("alpha_inv", 0.05))
    valid = l2[l2["p_inv_work"] >= alpha_inv]
    if valid.empty:
        # If all fail invariance screen, keep top by p-value to avoid degenerate fallback.
        valid = l2.sort_values(["p_inv_work", "rmse_work"], ascending=[False, True]).head(
            max(3, int(search_cfg.get("top_k", 12) // 3))
        )

    if forced_candidate_id is not None and str(forced_candidate_id).startswith("meta_"):
        raise RuntimeError("forced_candidate_id with meta_* is forbidden for EIDOS strict mode.")

    if forced_candidate_id:
        forced_rows = l2[l2["candidate_id"] == str(forced_candidate_id)]
        if forced_rows.empty:
            best_row = valid.sort_values("dl_total").iloc[0]
        else:
            best_row = forced_rows.iloc[0]
    elif ablation.conformal_in_inner_loop and not df_cal.empty:
        # Anti-pattern ablation: query cal in inner loop for model choice.
        scores = []
        for _, row in valid.iterrows():
            model = row["model"]
            mu_cal, _ = model.predict_mu_sigma(df_cal[feature_cols].to_numpy(float))
            mu_test_tmp, _ = model.predict_mu_sigma(df_test[feature_cols].to_numpy(float))
            cp = split_conformal_regression(
                y_cal=df_cal[target_col].to_numpy(float),
                yhat_cal=mu_cal,
                y_test=df_test[target_col].to_numpy(float),
                yhat_test=mu_test_tmp,
                alpha=0.1,
            )
            scores.append((cp.avg_length, -cp.coverage, row["candidate_id"]))
        scores.sort()
        best_id = scores[0][2]
        best_row = valid[valid["candidate_id"] == best_id].iloc[0]
    else:
        if selection_policy == "rmse_then_dl":
            best_row = valid.sort_values(["rmse_work", "dl_total", "p_inv_work"], ascending=[True, True, False]).iloc[0]
        elif selection_policy == "cv_rmse_then_dl":
            tmp = valid.copy()
            tmp["rmse_select"] = tmp["rmse_crossfit_work"].where(
                np.isfinite(tmp["rmse_crossfit_work"]), tmp["rmse_work"]
            )
            best_row = tmp.sort_values(["rmse_select", "rmse_work", "dl_total", "p_inv_work"], ascending=[True, True, True, False]).iloc[0]
        elif selection_policy == "cv_pooled_rmse_then_dl":
            tmp = valid.copy()
            tmp["rmse_crossfit_finite"] = tmp["rmse_crossfit_work"].where(
                np.isfinite(tmp["rmse_crossfit_work"]), tmp["rmse_work"]
            )
            tmp["rmse_select"] = _pooled_out_of_sample_rmse(
                tmp["rmse_crossfit_finite"],
                tmp["rmse_work"],
                n_crossfit=int(df_work.shape[0]),
                n_diagnosis=int(df_diag.shape[0]),
            )
            best_row = tmp.sort_values(
                ["rmse_select", "rmse_crossfit_finite", "rmse_work", "dl_total", "p_inv_work"],
                ascending=[True, True, True, True, False],
            ).iloc[0]
        elif selection_policy == "cv_rmse_pairwise_guarded":
            tmp = valid.copy()
            tmp["rmse_select"] = tmp["rmse_crossfit_work"].where(
                np.isfinite(tmp["rmse_crossfit_work"]), tmp["rmse_work"]
            )
            ranked = tmp.sort_values(
                ["rmse_select", "rmse_work", "dl_total", "p_inv_work"], ascending=[True, True, True, False]
            )
            best = ranked.iloc[0]
            pairwise_id = str(search_cfg.get("guarded_pairwise_candidate_id", "struct_dense_rbf_pairwise"))
            if str(best["candidate_id"]) == pairwise_id:
                fallback = ranked[ranked["candidate_id"] != pairwise_id]
                if not fallback.empty:
                    fb = fallback.iloc[0]
                    min_abs = float(search_cfg.get("guarded_pairwise_min_abs_gain", 0.0))
                    min_rel = float(search_cfg.get("guarded_pairwise_min_rel_gain", 0.02))
                    gain = float(fb["rmse_select"] - best["rmse_select"])
                    rel_gain = gain / max(float(fb["rmse_select"]), 1e-8)
                    best_row = best if (gain >= min_abs and rel_gain >= min_rel) else fb
                else:
                    best_row = best
            else:
                best_row = best
        elif selection_policy == "cv_rank_hybrid":
            tmp = valid.copy()
            tmp["rmse_select"] = tmp["rmse_crossfit_work"].where(
                np.isfinite(tmp["rmse_crossfit_work"]), tmp["rmse_work"]
            )
            tmp["r_cv"] = tmp["rmse_select"].rank(method="average")
            tmp["r_train"] = tmp["rmse_work"].rank(method="average")
            tmp["r_dl"] = tmp["dl_total"].rank(method="average")
            tmp["r_inv"] = (-tmp["p_inv_work"]).rank(method="average")
            tmp["r_mean"] = 0.55 * tmp["r_cv"] + 0.25 * tmp["r_train"] + 0.15 * tmp["r_dl"] + 0.05 * tmp["r_inv"]
            best_row = tmp.sort_values(["r_mean", "rmse_select", "rmse_work", "dl_total"], ascending=[True, True, True, True]).iloc[0]
        elif selection_policy == "hybrid_rank":
            tmp = valid.copy()
            tmp["r_rmse"] = tmp["rmse_work"].rank(method="average")
            tmp["r_nll"] = tmp["nll_work"].rank(method="average")
            tmp["r_dl"] = tmp["dl_total"].rank(method="average")
            tmp["r_inv"] = (-tmp["p_inv_work"]).rank(method="average")
            tmp["r_mean"] = 0.4 * tmp["r_rmse"] + 0.2 * tmp["r_nll"] + 0.3 * tmp["r_dl"] + 0.1 * tmp["r_inv"]
            best_row = tmp.sort_values("r_mean").iloc[0]
        elif selection_policy == "dl_only_bin_sensitive":
            best_row = valid.sort_values(
                ["dl_data", "dl_total", "dl_struct", "dl_param", "rmse_work", "p_inv_work"],
                ascending=[True, True, True, True, True, False],
            ).iloc[0]
        elif selection_policy == "dl_only_fisher_sensitive":
            best_row = valid.sort_values(
                ["dl_param", "dl_total", "dl_struct", "dl_data", "rmse_work", "p_inv_work"],
                ascending=[True, True, True, True, True, False],
            ).iloc[0]
        else:
            best_row = valid.sort_values("dl_total").iloc[0]

    ridge_multipliers_raw = search_cfg.get("postselect_ridge_multipliers", [])
    if isinstance(ridge_multipliers_raw, (int, float)):
        ridge_multipliers = [float(ridge_multipliers_raw)]
    elif isinstance(ridge_multipliers_raw, list):
        ridge_multipliers = sorted(
            {float(value) for value in ridge_multipliers_raw if np.isfinite(float(value)) and float(value) > 0.0}
        )
    else:
        ridge_multipliers = []
    if strict_protocol and len(ridge_multipliers) > 1:
        base_spec = best_row["model"].spec
        ridge_rows: list[dict[str, Any]] = []
        for multiplier in ridge_multipliers:
            ridge_value = max(1.0e-8, float(base_spec.ridge) * multiplier)
            variant = replace(base_spec, ridge=ridge_value)
            out = _fit_score_single(
                spec=variant,
                df_fit=df_work,
                df_eval=df_diag,
                feature_cols=feature_cols,
                target_col=target_col,
                delta_y=delta_y,
                use_bin_likelihood=ablation.use_bin_likelihood,
                use_fisher=False,
                fisher_probe_points=int(search_cfg.get("fisher_probe_points", 256)),
                fisher_jitter=float(search_cfg.get("fisher_jitter", 1e-4)),
                robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
                device=device,
                max_fisher_params=int(search_cfg.get("max_fisher_params", 28)),
                fisher_seed=_stable_seed(seed, variant.candidate_id, salt=10_003 + int(round(multiplier * 1000))),
                strict_param_code=True,
                fixed_param_step=float(search_cfg.get("fixed_param_step", 1e-3)),
            )
            cf = crossfit_residuals(
                df=df_work,
                feature_cols=feature_cols,
                target_col=target_col,
                env_col=env_col,
                model_factory=lambda spec=variant: LinearBasisModel(
                    spec, feature_cols, ridge=spec.ridge, device=device
                ),
                k=int(search_cfg.get("crossfit_k", 2)),
                seed=seed,
            )
            cv_rmse = _rmse(df_work[target_col].to_numpy(float), cf.mu)
            pooled_rmse = float(
                _pooled_out_of_sample_rmse(
                    np.asarray([cv_rmse]),
                    np.asarray([float(out["rmse_eval"])]),
                    n_crossfit=int(df_work.shape[0]),
                    n_diagnosis=int(df_diag.shape[0]),
                )[0]
            )
            mu_diag, sigma_diag = out["model"].predict_mu_sigma(df_diag[feature_cols].to_numpy(float))
            z_diag = (df_diag[target_col].to_numpy(float) - mu_diag) / np.maximum(sigma_diag, 1e-8)
            z_diag = normalize_residuals_by_env(
                z_diag,
                df_diag[env_col].to_numpy(int),
                mode=str(inv_cfg.get("residual_normalization", "global")),
                min_scale=float(inv_cfg.get("residual_norm_min_scale", 1e-6)),
                shrinkage=float(inv_cfg.get("residual_norm_shrinkage", 4.0)),
            )
            ridge_inv = mmd_rff_permutation_test(
                z=z_diag,
                env=df_diag[env_col].to_numpy(int),
                n_features=int(inv_cfg.get("rff_features", 128)),
                n_perm=int(inv_cfg.get("work_permutations", 100)),
                seed=seed,
            )
            ridge_rows.append(
                {
                    **out,
                    "ridge": ridge_value,
                    "ridge_multiplier": multiplier,
                    "rmse_work": float(out["rmse_eval"]),
                    "rmse_crossfit_work": cv_rmse,
                    "pooled_oos_rmse": pooled_rmse,
                    "nll_work": float(out["nll_eval"]),
                    "p_inv_work": float(ridge_inv.p_value),
                    "inv_stat_work": float(ridge_inv.stat),
                }
            )
        ridge_table = pd.DataFrame(ridge_rows)
        ridge_valid = ridge_table[ridge_table["p_inv_work"] >= alpha_inv]
        if ridge_valid.empty:
            ridge_valid = ridge_table.sort_values(
                ["p_inv_work", "pooled_oos_rmse"], ascending=[False, True]
            ).head(1)
        tuned = ridge_valid.sort_values(
            ["pooled_oos_rmse", "rmse_crossfit_work", "rmse_work", "dl_total"]
        ).iloc[0]
        if ablation.use_fisher:
            verified = _fit_score_single(
                spec=tuned["model"].spec,
                df_fit=df_work,
                df_eval=df_diag,
                feature_cols=feature_cols,
                target_col=target_col,
                delta_y=delta_y,
                use_bin_likelihood=ablation.use_bin_likelihood,
                use_fisher=True,
                fisher_probe_points=int(search_cfg.get("fisher_probe_points", 256)),
                fisher_jitter=float(search_cfg.get("fisher_jitter", 1e-4)),
                robust_eps=float(search_cfg.get("robust_mixture_eps", 1e-4)),
                device=device,
                max_fisher_params=int(search_cfg.get("max_fisher_params", 28)),
                fisher_seed=int(tuned["fisher_probe_seed"]),
                strict_param_code=True,
                fixed_param_step=float(search_cfg.get("fixed_param_step", 1e-3)),
            )
            tuned = tuned.copy()
            for key, value in verified.items():
                tuned[key] = value
        best_row = best_row.copy()
        for key in (
            "model",
            "dl_total",
            "dl_grammar",
            "dl_struct",
            "dl_param",
            "dl_data",
            "p_inv_work",
            "inv_stat_work",
            "fisher_logdet",
            "fisher_jitter",
            "fisher_eig_min",
            "fisher_eig_max",
            "fisher_cond_number",
            "fisher_effective_rank",
            "fisher_grad_norm_mean",
            "fisher_grad_norm_std",
            "fisher_grad_norm_max",
            "fisher_probe_seed",
            "fisher_probe_count",
            "rmse_work",
            "rmse_crossfit_work",
            "nll_work",
        ):
            best_row[key] = tuned[key]
        best_row["postselect_ridge"] = float(tuned["ridge"])
        best_row["postselect_ridge_multiplier"] = float(tuned["ridge_multiplier"])
        best_row["postselect_pooled_oos_rmse"] = float(tuned["pooled_oos_rmse"])
        ledger.log(
            {
                "run_id": run_id,
                "git_hash": git_hash,
                "seed": seed,
                "dataset_id": dataset_id,
                "env_id": "all",
                "split": "diag",
                "delta_y_source": "metadata",
                "delta_y_value": delta_y,
                "model_id": model_id,
                "candidate_id": str(best_row["candidate_id"]),
                "dl_total": float(best_row["dl_total"]),
                "p_inv_work": float(best_row["p_inv_work"]),
                "failure_reason": None,
                "notes": (
                    "postselect_ridge_calibration "
                    f"ridge={float(tuned['ridge']):.12g} multiplier={float(tuned['ridge_multiplier']):.6g} "
                    f"pooled_oos_rmse={float(tuned['pooled_oos_rmse']):.12g}"
                ),
            }
        )

    selection_model = best_row["model"]
    best_model = selection_model
    best_candidate_id = str(best_row["candidate_id"])
    final_refit_roles = discovery_role
    final_refit_n = int(df_work.shape[0])
    if strict_protocol:
        df_refit = pd.concat([df_work, df_diag], ignore_index=True)
        best_model = LinearBasisModel(
            spec=selection_model.spec,
            feature_cols=feature_cols,
            ridge=selection_model.ridge,
            device=device,
        )
        best_model.fit(
            df_refit[feature_cols].to_numpy(float),
            df_refit[target_col].to_numpy(float),
        )
        quantized = quantize_parameters(
            best_model.param_vector(),
            steps=float(search_cfg.get("fixed_param_step", 1e-3)),
        )
        best_model.set_param_vector(quantized.values)
        final_refit_roles = "disc+diag"
        final_refit_n = int(df_refit.shape[0])

    alpha = _primary_conformal_alpha(conf_cfg)
    if best_candidate_id in lock_cert_cache and not strict_protocol:
        p_inv_lock = float(lock_cert_cache[best_candidate_id])
        inv_stat_lock = float(lock_stat_cache.get(best_candidate_id, np.nan))
    else:
        lock_res = _evaluate_lock_invariance_result(
            model=best_model,
            df_lock=df_lock,
            feature_cols=feature_cols,
            target_col=target_col,
            env_col=env_col,
            inv_cfg=inv_cfg,
            seed=_stable_seed(seed, best_candidate_id, salt=211),
        )
        p_inv_lock = float(lock_res.p_value)
        inv_stat_lock = float(lock_res.stat)
        lock_cert_cache[best_candidate_id] = p_inv_lock
        lock_stat_cache[best_candidate_id] = inv_stat_lock

    if best_candidate_id in conf_cert_cache and not strict_protocol:
        cp_coverage, cp_avg_length = conf_cert_cache[best_candidate_id]
    else:
        cp_coverage, cp_avg_length = _evaluate_conformal_certificate(
            model=best_model,
            df_cal=df_cal,
            df_test=df_test,
            feature_cols=feature_cols,
            target_col=target_col,
            alpha=alpha,
        )
        conf_cert_cache[best_candidate_id] = (cp_coverage, cp_avg_length)

    final_cp = _evaluate_conformal_result(
        model=best_model,
        df_cal=df_cal,
        df_test=df_test,
        feature_cols=feature_cols,
        target_col=target_col,
        alpha=alpha,
    )
    cp_coverage = float(final_cp.coverage)
    cp_avg_length = float(final_cp.avg_length)

    mu_test, sigma_test = best_model.predict_mu_sigma(df_test[feature_cols].to_numpy(float))

    rmse_test = _rmse(df_test[target_col].to_numpy(float), mu_test)
    nll_test = _nll_student(df_test[target_col].to_numpy(float), mu_test, sigma_test)

    # ledger for final event
    ledger.log(
        {
            "run_id": run_id,
            "git_hash": git_hash,
            "seed": seed,
            "dataset_id": dataset_id,
            "env_id": "all",
            "split": "test",
            "delta_y_source": "metadata",
            "delta_y_value": delta_y,
            "model_id": model_id,
            "candidate_id": best_candidate_id,
            "dl_grammar": float(best_row.get("dl_grammar", 0.0)),
            "dl_struct": float(best_row["dl_struct"]),
            "dl_param": float(best_row["dl_param"]),
            "dl_data": float(best_row["dl_data"]),
            "dl_total": float(best_row["dl_total"]),
            "inv_test_type": "mmd_rff",
            "p_inv_work": float(best_row["p_inv_work"]),
            "inv_stat_work": float(best_row.get("inv_stat_work", np.nan)),
            "p_inv_lock": p_inv_lock,
            "inv_stat_lock": inv_stat_lock,
            "conformal_coverage": float(cp_coverage) if np.isfinite(cp_coverage) else None,
            "conformal_length": float(cp_avg_length) if np.isfinite(cp_avg_length) else None,
            "failure_reason": None,
            "notes": f"final_candidate refit={final_refit_roles}",
            "fisher_logdet": float(best_row["fisher_logdet"]) if np.isfinite(best_row["fisher_logdet"]) else None,
            "fisher_jitter": float(best_row["fisher_jitter"]) if np.isfinite(best_row["fisher_jitter"]) else None,
            "fisher_eig_min": float(best_row["fisher_eig_min"]) if np.isfinite(best_row["fisher_eig_min"]) else None,
            "fisher_eig_max": float(best_row["fisher_eig_max"]) if np.isfinite(best_row["fisher_eig_max"]) else None,
            "fisher_cond_number": float(best_row["fisher_cond_number"])
            if np.isfinite(best_row["fisher_cond_number"])
            else None,
            "fisher_effective_rank": float(best_row["fisher_effective_rank"])
            if np.isfinite(best_row["fisher_effective_rank"])
            else None,
            "fisher_grad_norm_mean": float(best_row["fisher_grad_norm_mean"])
            if np.isfinite(best_row["fisher_grad_norm_mean"])
            else None,
            "fisher_probe_seed": int(best_row["fisher_probe_seed"]),
            "fisher_probe_count": int(best_row["fisher_probe_count"]),
        }
    )

    # top-k report
    topk_df = _topk_diverse(
        ranked_df=valid.sort_values("dl_total"),
        top_k=int(search_cfg.get("top_k", 12)),
        df_work=df_work,
        feature_cols=feature_cols,
        seed=_stable_seed(seed, dataset_id, salt=503),
    )
    topk_records: list[dict[str, Any]] = []
    topk_feature_bank: list[dict[str, Any]] = []
    for _, row in topk_df.iterrows():
        cid = str(row["candidate_id"])
        if cid not in lock_cert_cache and (not strict_protocol or cid == best_candidate_id):
            lock_res = _evaluate_lock_invariance_result(
                model=row["model"],
                df_lock=df_lock,
                feature_cols=feature_cols,
                target_col=target_col,
                env_col=env_col,
                inv_cfg=inv_cfg,
                seed=_stable_seed(seed, cid, salt=271),
            )
            lock_cert_cache[cid] = float(lock_res.p_value)
            lock_stat_cache[cid] = float(lock_res.stat)
        topk_records.append(
            {
                "candidate_id": cid,
                "dl_total": float(row["dl_total"]),
                "p_inv_work": float(row["p_inv_work"]),
                "inv_stat_work": float(row.get("inv_stat_work", np.nan)),
                "p_inv_lock": float(lock_cert_cache[cid]) if cid in lock_cert_cache else float("nan"),
                "inv_stat_lock": float(lock_stat_cache.get(cid, np.nan)),
                "dl_grammar": float(row["dl_grammar"]),
                "dl_struct": float(row["dl_struct"]),
                "dl_param": float(row["dl_param"]),
                "dl_data": float(row["dl_data"]),
                "rmse_work": float(row["rmse_work"]),
                "nll_work": float(row["nll_work"]),
            }
        )
        topk_feature_bank.append({"candidate_id": cid, "model": row["model"]})

    total_runtime = time.time() - t0
    layer_stats = {
        "n_candidates_generated": screen.n_before,
        "n_candidates_after_screen": screen.n_after,
        "n_candidates_rejected": len(screen.rejected),
        "n_layer2": len(layer2_specs),
        "n_topk": len(topk_records),
        "layer1_time_sec": layer1_time,
        "layer2_time_sec": layer2_time,
        "runtime_sec": total_runtime,
        "final_refit_roles": final_refit_roles,
        "final_refit_n": final_refit_n,
    }
    if audit_trace:
        try:
            expr_str = best_model.to_expr().to_string()
        except Exception:
            expr_str = ""
        layer_stats.update(
            {
                "final_candidate_id": best_candidate_id,
                "final_candidate_expr": expr_str,
                "final_dl_grammar": float(best_row.get("dl_grammar", 0.0)),
                "final_dl_struct": float(best_row["dl_struct"]),
                "final_dl_param": float(best_row["dl_param"]),
                "final_dl_data": float(best_row["dl_data"]),
                "final_dl_total": float(best_row["dl_total"]),
                "final_rmse_work": float(best_row["rmse_work"]),
                "final_rmse_crossfit_work": float(best_row.get("rmse_crossfit_work", np.nan))
                if np.isfinite(best_row.get("rmse_crossfit_work", np.nan))
                else None,
                "final_postselect_ridge": float(best_row.get("postselect_ridge", selection_model.ridge)),
                "final_postselect_ridge_multiplier": float(best_row.get("postselect_ridge_multiplier", 1.0)),
                "final_postselect_pooled_oos_rmse": float(best_row.get("postselect_pooled_oos_rmse", np.nan))
                if np.isfinite(best_row.get("postselect_pooled_oos_rmse", np.nan))
                else None,
                "final_nll_work": float(best_row["nll_work"]),
                "final_inv_stat_work": float(best_row.get("inv_stat_work", np.nan)),
                "final_inv_stat_lock": float(inv_stat_lock),
                "target_unit_signature": _format_unit_signature(target_unit_map),
                "feature_unit_signature": json.dumps(
                    {str(k): _format_unit_signature(v) for k, v in unit_map.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return EngineResult(
        best_candidate_id=best_candidate_id,
        rmse_test=rmse_test,
        nll_test=nll_test,
        dl_total=float(best_row["dl_total"]),
        p_inv_work=float(best_row["p_inv_work"]),
        p_inv_lock=p_inv_lock,
        conformal_coverage=float(cp_coverage) if np.isfinite(cp_coverage) else float("nan"),
        conformal_length=float(cp_avg_length) if np.isfinite(cp_avg_length) else float("nan"),
        topk=topk_records,
        runtime_sec=total_runtime,
        layer_stats=layer_stats,
        best_model=best_model,
        topk_feature_bank=topk_feature_bank,
        test_sample_ids=df_test["sample_id"].to_numpy(copy=True)
        if "sample_id" in df_test.columns
        else df_test.index.to_numpy(copy=True),
        test_y=df_test[target_col].to_numpy(float, copy=True),
        test_pred=np.asarray(mu_test, dtype=float).copy(),
        test_lower=final_cp.lower,
        test_upper=final_cp.upper,
    )
