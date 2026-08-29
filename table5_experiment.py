from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.eidos.dsl.grammar import Const, Expr, Var, op1, op2, op3
from src.eidos.invariance.crossfit import crossfit_residuals
from src.eidos.protocol.strict import five_role_split_by_env
from src.eidos.search.engine import _fit_score_single
from src.eidos.search.models import CandidateSpec, LinearBasisModel


BANK_NAME = "physics_template_bank_v2"
BANK_SIZE = 240
NOISE_SIGMA = 0.80
REGISTERED_SEEDS = (20260225, 20260226, 20260227, 20260228, 20260229)
DISPLAY_CASES = ("oscillator", "lorentz", "blackbody")
ALL_BANK_FAMILIES = ("oscillator", "lorentz", "relativistic", "kepler", "blackbody")

CASE_NAMES = {
    "oscillator": "Oscillator",
    "lorentz": "Lorentz",
    "blackbody": "Blackbody proxy",
}
DISPLAY_TERMS = {
    "oscillator": (r"\sin(1.1x_1)", r"\cos(x_2)", r"x_3"),
    "lorentz": (r"\frac{x_1x_2}{1+x_3^2}",),
    "blackbody": (
        r"\frac{x_1^3}{\exp(\operatorname{clip}(x_2,0,5))-1+0.01}",
        r"x_3",
    ),
}
TRUTH_FORMULAS = {
    "oscillator": r"2.2\sin(1.1x_1)+0.8\cos(x_2)-0.2x_3",
    "lorentz": r"\frac{x_1x_2}{1+x_3^2}",
    "blackbody": (
        r"\frac{x_1^3}{\exp(\operatorname{clip}(x_2,0,5))-1+0.01}+0.05x_3"
    ),
}


@dataclass(frozen=True)
class BankEntry:
    spec: CandidateSpec
    template_family: str
    truth_for: str
    distractor_class: str


def _cube(expr: Expr) -> Expr:
    return op2("mul", op2("mul", expr, expr), expr)


def _square(expr: Expr) -> Expr:
    return op2("mul", expr, expr)


def _clip(expr: Expr, low: float, high: float) -> Expr:
    return op3("clip", expr, Const(low), Const(high))


def build_candidate_bank(
    bank_size: int = BANK_SIZE,
    seed: int = 20260822,
) -> tuple[list[BankEntry], dict[str, str]]:
    """Rebuild the task-independent candidate bank used by Table 5."""

    variables = ("x1", "x2", "x3")
    staged: dict[str, tuple[list[Expr], str, str, str]] = {}

    def add(
        basis: list[Expr],
        family: str,
        truth_for: str = "",
        distractor: str = "near_equivalent",
    ) -> None:
        signature = json.dumps([expr.to_string() for expr in basis], separators=(",", ":"))
        if signature not in staged or truth_for:
            staged[signature] = (basis, family, truth_for, distractor)

    for first, second, third in permutations(variables, 3):
        x1, x2, x3 = Var(first), Var(second), Var(third)
        for frequency in (0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8):
            for second_op in ("cos", "sin"):
                basis = [
                    Const(1.0),
                    op1("sin", op2("mul", Const(frequency), x1)),
                    op1(second_op, x2),
                    x3,
                ]
                exact = (
                    first == "x1"
                    and second == "x2"
                    and third == "x3"
                    and frequency == 1.1
                    and second_op == "cos"
                )
                add(
                    basis,
                    "oscillator",
                    "oscillator" if exact else "",
                    "frequency_or_operator",
                )

        for denominator_power in (1, 2, 3, 4):
            for denominator_offset in (0.5, 1.0, 2.0):
                denominator_term = x3
                for _ in range(denominator_power - 1):
                    denominator_term = op2("mul", denominator_term, x3)
                basis = [
                    Const(1.0),
                    op2(
                        "div",
                        op2("mul", x1, x2),
                        op2("add", Const(denominator_offset), denominator_term),
                    ),
                ]
                exact = (
                    first == "x1"
                    and second == "x2"
                    and third == "x3"
                    and denominator_power == 2
                    and denominator_offset == 1.0
                )
                add(
                    basis,
                    "lorentz",
                    "lorentz" if exact else "",
                    "variable_power_or_offset",
                )

        for beta in (0.06, 0.08, 0.09765625, 0.12, 0.16):
            for exponent in (-1.0, -0.5):
                relativistic = op2(
                    "mul",
                    x1,
                    op2(
                        "pow",
                        _clip(
                            op2("sub", Const(1.0), op2("mul", Const(beta), _square(x2))),
                            0.06,
                            4.0,
                        ),
                        Const(exponent),
                    ),
                )
                basis = [Const(1.0), relativistic, x3]
                exact = (
                    first == "x1"
                    and second == "x2"
                    and third == "x3"
                    and beta == 0.09765625
                    and exponent == -0.5
                )
                add(
                    basis,
                    "relativistic",
                    "relativistic" if exact else "",
                    "coefficient_exponent_or_variable",
                )

    for first, second in permutations(variables, 2):
        numerator, denominator = Var(first), Var(second)
        for denominator_power in (2, 3, 4):
            denominator_term = denominator
            for _ in range(denominator_power - 1):
                denominator_term = op2("mul", denominator_term, denominator)
            for exponent in (1.0 / 3.0, 0.5, 2.0 / 3.0):
                basis = [
                    Const(1.0),
                    op2(
                        "pow",
                        op2(
                            "div",
                            _clip(numerator, 0.1, 10.0),
                            _clip(denominator_term, 0.05, 100.0),
                        ),
                        Const(exponent),
                    ),
                ]
                exact = (
                    first == "x1"
                    and second == "x2"
                    and denominator_power == 3
                    and exponent == 0.5
                )
                add(
                    basis,
                    "kepler",
                    "kepler" if exact else "",
                    "power_exponent_or_variable",
                )

    for first, second, third in permutations(variables, 3):
        numerator, exponent_var, additive = Var(first), Var(second), Var(third)
        for numerator_power in (2, 3, 4):
            numerator_term = numerator
            for _ in range(numerator_power - 1):
                numerator_term = op2("mul", numerator_term, numerator)
            for exponent_scale in (0.8, 1.0, 1.2):
                denominator = op2(
                    "add",
                    op2(
                        "sub",
                        op1(
                            "exp",
                            _clip(
                                op2("mul", Const(exponent_scale), exponent_var),
                                0.0,
                                5.0,
                            ),
                        ),
                        Const(1.0),
                    ),
                    Const(1.0e-2),
                )
                basis = [Const(1.0), op2("div", numerator_term, denominator), additive]
                exact = (
                    first == "x1"
                    and second == "x2"
                    and third == "x3"
                    and numerator_power == 3
                    and exponent_scale == 1.0
                )
                add(
                    basis,
                    "blackbody",
                    "blackbody" if exact else "",
                    "power_scale_or_variable",
                )

    exact_rows = [(signature, payload) for signature, payload in staged.items() if payload[2]]
    distractors = [(signature, payload) for signature, payload in staged.items() if not payload[2]]
    if len(exact_rows) != len(ALL_BANK_FAMILIES):
        raise AssertionError(f"expected five truth templates, found {len(exact_rows)}")
    if bank_size < 200 or bank_size > len(staged):
        raise ValueError(f"bank_size must be in [200, {len(staged)}]")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(distractors), size=bank_size - len(exact_rows), replace=False)
    selected = exact_rows + [distractors[int(index)] for index in chosen_idx]
    rng.shuffle(selected)

    entries: list[BankEntry] = []
    truth_ids: dict[str, str] = {}
    for offset, (_, payload) in enumerate(selected, start=401):
        basis, family, truth_for, distractor = payload
        candidate_id = f"cand_{offset:04d}"
        entry = BankEntry(
            spec=CandidateSpec(
                candidate_id=candidate_id,
                basis_exprs=basis,
                sigma_mode="constant",
                ridge=1.0e-8,
            ),
            template_family=family,
            truth_for=truth_for,
            distractor_class="truth" if truth_for else distractor,
        )
        entries.append(entry)
        if truth_for:
            truth_ids[truth_for] = candidate_id
    return entries, truth_ids


def generate_case(
    generator: str,
    sigma: float,
    seed: int,
    n_per_env: int = 120,
) -> pd.DataFrame:
    """Generate role-labelled observations directly from the declared mechanism."""

    if generator not in DISPLAY_CASES:
        raise ValueError(f"Table 5 does not contain generator: {generator}")
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for env in range(5):
        x1 = rng.uniform(0.2 + 0.05 * env, 2.5 + 0.3 * env, size=n_per_env)
        x2 = rng.uniform(0.1, 2.0 + 0.2 * env, size=n_per_env)
        x3 = rng.normal(0.0, 1.0 + 0.15 * env, size=n_per_env)
        if generator == "oscillator":
            truth = 2.2 * np.sin(1.1 * x1) + 0.8 * np.cos(x2) - 0.2 * x3
        elif generator == "lorentz":
            truth = (x1 * x2) / (1.0 + x3**2)
        else:
            truth = (
                (x1**3) / (np.exp(np.clip(x2, 0.0, 5.0)) - 1.0 + 1.0e-2)
                + 0.05 * x3
            )
        y_tilde = np.round((truth + rng.normal(0.0, sigma, size=n_per_env)) / 1.0e-3) * 1.0e-3
        rows.append(
            pd.DataFrame(
                {
                    "env_id": env,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "truth": truth,
                    "y_tilde": y_tilde,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame["split"] = five_role_split_by_env(frame["env_id"].to_numpy(), seed=seed)
    return frame


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def select_and_refit(
    frame: pd.DataFrame,
    bank: list[BankEntry],
    seed: int,
    crossfit_k: int = 3,
) -> tuple[BankEntry, LinearBasisModel, dict[str, float]]:
    """Select on discovery/diagnosis and refit on those same allowed roles."""

    features = ["x1", "x2", "x3"]
    disc = frame[frame["split"] == "disc"].copy()
    diag = frame[frame["split"] == "diag"].copy()
    allowed = frame[frame["split"].isin(["disc", "diag"])].copy()
    if disc.empty or diag.empty or allowed.shape[0] != disc.shape[0] + diag.shape[0]:
        raise ValueError("selection requires disjoint, non-empty discovery and diagnosis roles")

    scored_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(bank):
        holdout = _fit_score_single(
            spec=entry.spec,
            df_fit=disc,
            df_eval=diag,
            feature_cols=features,
            target_col="y_tilde",
            delta_y=1.0e-3,
            use_bin_likelihood=True,
            use_fisher=False,
            fisher_probe_points=16,
            fisher_jitter=1.0e-4,
            robust_eps=1.0e-4,
            device="cpu",
            max_fisher_params=16,
            fisher_seed=seed + index,
            strict_param_code=True,
            fixed_param_step=1.0e-3,
        )
        crossfit = crossfit_residuals(
            df=allowed,
            feature_cols=features,
            target_col="y_tilde",
            env_col="env_id",
            model_factory=lambda spec=entry.spec: LinearBasisModel(
                spec=spec,
                feature_cols=features,
                ridge=spec.ridge,
                device="cpu",
            ),
            k=crossfit_k,
            seed=seed,
        )
        scored_rows.append(
            {
                "entry": entry,
                "dl_total": float(holdout["dl_total"]),
                "diag_rmse": float(holdout["rmse_eval"]),
                "cv_rmse": _rmse(allowed["y_tilde"].to_numpy(float), crossfit.mu),
            }
        )

    scored = pd.DataFrame(scored_rows)
    best_index = int(scored.sort_values(["cv_rmse", "diag_rmse", "dl_total"]).index[0])
    best_row = scored.loc[best_index]
    best_entry = best_row["entry"]

    refit = _fit_score_single(
        spec=best_entry.spec,
        df_fit=allowed,
        df_eval=allowed,
        feature_cols=features,
        target_col="y_tilde",
        delta_y=1.0e-3,
        use_bin_likelihood=True,
        use_fisher=False,
        fisher_probe_points=16,
        fisher_jitter=1.0e-4,
        robust_eps=1.0e-4,
        device="cpu",
        max_fisher_params=16,
        fisher_seed=seed + len(bank),
        strict_param_code=True,
        fixed_param_step=1.0e-3,
    )
    diagnostics = {
        "selected_dl_total": float(best_row["dl_total"]),
        "selected_diag_rmse": float(best_row["diag_rmse"]),
        "selected_cv_rmse": float(best_row["cv_rmse"]),
    }
    return best_entry, refit["model"], diagnostics


def _signed_term(coefficient: float, term: str) -> str:
    return f" {'-' if coefficient < 0 else '+'} {abs(float(coefficient)):.4g}{term}"


def format_eidos_formula(model: LinearBasisModel, generator: str) -> str:
    """Convert the standardized linear-basis fit back to paper coordinates."""

    if model.w is None or model.phi_mean is None or model.phi_std is None:
        raise RuntimeError("EIDOS model is not fitted")
    means = np.asarray(model.phi_mean, dtype=float).reshape(-1)
    scales = np.asarray(model.phi_std, dtype=float).reshape(-1)
    weights = np.asarray(model.w, dtype=float).reshape(-1)
    coefficients = weights / scales
    intercept = -float(np.sum(weights * means / scales)) + float(coefficients[0])
    parts = [f"{intercept:.4g}"]
    for coefficient, term in zip(coefficients[1:], DISPLAY_TERMS[generator], strict=True):
        parts.append(_signed_term(float(coefficient), term))
    return "".join(parts)


def run_cell(
    generator: str,
    seed: int,
    bank: list[BankEntry],
    truth_ids: dict[str, str],
    noise_sigma: float = NOISE_SIGMA,
) -> dict[str, Any]:
    started = time.perf_counter()
    frame = generate_case(generator, noise_sigma, seed)
    best_entry, model, diagnostics = select_and_refit(frame, bank, seed)
    expected_id = truth_ids[generator]
    if best_entry.spec.candidate_id != expected_id:
        raise AssertionError(
            f"exact-recovery gate failed for {generator}, seed {seed}: "
            f"selected {best_entry.spec.candidate_id}, expected {expected_id}"
        )

    test = frame[frame["split"] == "test"]
    x_test = test[["x1", "x2", "x3"]].to_numpy(float)
    prediction, _ = model.predict_mu_sigma(x_test)
    truth = test["truth"].to_numpy(float)
    noisy = test["y_tilde"].to_numpy(float)
    scale = max(float(np.std(truth)), 1.0e-12)
    return {
        "case": CASE_NAMES[generator],
        "generator": generator,
        "noise_sigma": noise_sigma,
        "seed": seed,
        "bank_name": BANK_NAME,
        "bank_size": len(bank),
        "selected_candidate_id": best_entry.spec.candidate_id,
        "expected_candidate_id": expected_id,
        "exact_generating_structure": True,
        "truth_formula": TRUTH_FORMULAS[generator],
        "eidos_formula": format_eidos_formula(model, generator),
        "semantic_nrmse": _rmse(truth, prediction) / scale,
        "semantic_max_scaled_error": float(np.max(np.abs(truth - prediction)) / scale),
        "test_rmse": _rmse(noisy, prediction),
        "fit_roles": "disc+diag",
        "selection_policy": "cv_rmse_then_dl",
        **diagnostics,
        "runtime_sec": time.perf_counter() - started,
    }


def candidate_bank_frame(bank: Iterable[BankEntry]) -> pd.DataFrame:
    rows = []
    for entry in bank:
        rows.append(
            {
                "bank_name": BANK_NAME,
                "candidate_id": entry.spec.candidate_id,
                "template_family": entry.template_family,
                "truth_for": entry.truth_for,
                "distractor_class": entry.distractor_class,
                "basis_preorder": json.dumps(
                    [expr.to_string() for expr in entry.spec.basis_exprs],
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def run_table5_experiment(
    seeds: Iterable[int] = REGISTERED_SEEDS,
    noise_sigma: float = NOISE_SIGMA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all EIDOS cells needed for the displayed records and five-seed gate."""

    registered = tuple(int(seed) for seed in seeds)
    if not registered:
        raise ValueError("at least one seed is required")
    bank, truth_ids = build_candidate_bank()
    rows: list[dict[str, Any]] = []
    for generator in DISPLAY_CASES:
        for seed in registered:
            print(f"[EIDOS] {generator:10s} sigma={noise_sigma:g} seed={seed}", flush=True)
            rows.append(run_cell(generator, seed, bank, truth_ids, noise_sigma=noise_sigma))
    all_runs = pd.DataFrame(rows)
    first_seed = registered[0]
    table5 = all_runs[all_runs["seed"] == first_seed][
        [
            "case",
            "generator",
            "noise_sigma",
            "seed",
            "truth_formula",
            "eidos_formula",
            "selected_candidate_id",
            "semantic_nrmse",
        ]
    ].reset_index(drop=True)
    gate = (
        all_runs.groupby(["case", "generator"], sort=False, as_index=False)
        .agg(
            recovered=("exact_generating_structure", "sum"),
            registered_runs=("exact_generating_structure", "size"),
            semantic_nrmse_mean=("semantic_nrmse", "mean"),
            semantic_nrmse_max=("semantic_nrmse", "max"),
        )
    )
    gate["gate"] = gate["recovered"].astype(str) + "/" + gate["registered_runs"].astype(str)
    return table5, all_runs, gate, candidate_bank_frame(bank)
