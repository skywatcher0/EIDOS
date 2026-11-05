from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

from experiments.interval_core.dsl.domain_safety import safe_eval
from experiments.interval_core.dsl.grammar import Const, Expr, Var, op1, op2, op3


def _softplus(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -20.0, 20.0)
    return np.log1p(np.exp(x_clip))


@dataclass
class CandidateSpec:
    candidate_id: str
    basis_exprs: list[Expr]
    sigma_mode: str = "constant"  # constant | linear_abs_x0
    noise: str = "student_t"
    df: float = 3.0
    ridge: float = 0.5
    parent_candidate_id: str | None = None


class LinearBasisModel:
    def __init__(
        self,
        spec: CandidateSpec,
        feature_cols: list[str],
        ridge: float = 0.5,
        device: str = "cpu",
    ):
        self.spec = spec
        self.feature_cols = feature_cols
        self.ridge = ridge
        self.device = device
        self.w: np.ndarray | None = None
        self.sigma_params: np.ndarray | None = None
        self.phi_mean: np.ndarray | None = None
        self.phi_std: np.ndarray | None = None

    def _var_map(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {c: x[:, i] for i, c in enumerate(self.feature_cols)}

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        vmap = self._var_map(x)
        cols = []
        for expr in self.spec.basis_exprs:
            out = safe_eval(expr, vmap)
            cols.append(np.asarray(out.values, dtype=float))
        phi = np.column_stack(cols)
        phi = np.nan_to_num(phi, nan=0.0, posinf=1e6, neginf=-1e6)
        return phi

    def _normalize_phi_fit(self, phi: np.ndarray) -> np.ndarray:
        self.phi_mean = phi.mean(axis=0, keepdims=True)
        self.phi_std = phi.std(axis=0, keepdims=True) + 1e-6
        # keep first basis (typically constant) unnormalized for interpretability
        if phi.shape[1] > 0:
            self.phi_mean[0, 0] = 0.0
            self.phi_std[0, 0] = 1.0
        return (phi - self.phi_mean) / self.phi_std

    def _normalize_phi_pred(self, phi: np.ndarray) -> np.ndarray:
        if self.phi_mean is None or self.phi_std is None:
            return phi
        return (phi - self.phi_mean) / self.phi_std

    def fit(self, x: np.ndarray, y: np.ndarray, env: np.ndarray | None = None) -> "LinearBasisModel":
        phi_raw = self._basis_matrix(x)
        phi = self._normalize_phi_fit(phi_raw)
        ridge_eff = float(max(1e-8, self.ridge))
        if self.device.startswith("cuda"):
            try:
                import torch

                phi_t = torch.as_tensor(phi, dtype=torch.float32, device=self.device)
                y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
                a_t = phi_t.T @ phi_t + ridge_eff * torch.eye(phi_t.shape[1], device=self.device)
                b_t = phi_t.T @ y_t
                w_t = torch.linalg.solve(a_t, b_t)
                self.w = w_t.detach().cpu().numpy().astype(float)
            except Exception:
                a = phi.T @ phi + ridge_eff * np.eye(phi.shape[1])
                b = phi.T @ y
                self.w = np.linalg.solve(a, b)
        else:
            a = phi.T @ phi + ridge_eff * np.eye(phi.shape[1])
            b = phi.T @ y
            self.w = np.linalg.solve(a, b)
        mu = phi @ self.w
        resid = y - mu

        if self.spec.sigma_mode == "constant":
            sigma0 = float(np.std(resid) + 1e-3)
            self.sigma_params = np.array([np.log(np.exp(sigma0) - 1.0)], dtype=float)
        else:
            x0 = np.abs(x[:, 0])
            z = np.column_stack([np.ones_like(x0), x0])
            target = np.log(np.abs(resid) + 1e-3)
            beta = np.linalg.lstsq(z, target, rcond=None)[0]
            self.sigma_params = beta.astype(float)
        return self

    def predict_mu_sigma(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.w is None or self.sigma_params is None:
            raise RuntimeError("Model not fitted.")
        phi = self._normalize_phi_pred(self._basis_matrix(x))
        mu = phi @ self.w
        if self.spec.sigma_mode == "constant":
            sigma = _softplus(np.full(mu.shape, self.sigma_params[0]))
        else:
            x0 = np.abs(x[:, 0])
            raw = self.sigma_params[0] + self.sigma_params[1] * x0
            sigma = _softplus(raw) + 1e-3
        return mu, sigma

    def param_vector(self) -> np.ndarray:
        if self.w is None or self.sigma_params is None:
            raise RuntimeError("Model not fitted.")
        return np.concatenate([self.w, self.sigma_params])

    def set_param_vector(self, p: np.ndarray) -> None:
        k = len(self.spec.basis_exprs)
        self.w = np.asarray(p[:k], dtype=float)
        self.sigma_params = np.asarray(p[k:], dtype=float)

    def logpdf_vector(self, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.set_param_vector(p)
        mu, sigma = self.predict_mu_sigma(x)
        z = (y - mu) / np.maximum(sigma, 1e-8)
        return t.logpdf(z, df=self.spec.df) - np.log(np.maximum(sigma, 1e-8))

    def to_expr(self) -> Expr:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        if self.phi_mean is None or self.phi_std is None:
            coefficients = np.asarray(self.w, dtype=float)
            offset = 0.0
        else:
            means = np.asarray(self.phi_mean, dtype=float).reshape(-1)
            scales = np.asarray(self.phi_std, dtype=float).reshape(-1)
            coefficients = np.asarray(self.w, dtype=float) / scales
            offset = -float(np.sum(np.asarray(self.w, dtype=float) * means / scales))
        # The fitted weights act on standardized basis columns.  Export the
        # algebraically equivalent expression in the original basis so formula
        # audits and downstream reuse see the model that actually predicted.
        terms: list[Expr] = [Const(offset)]
        for coefficient, basis in zip(coefficients, self.spec.basis_exprs):
            terms.append(op2("mul", Const(float(coefficient)), basis))
        cur = terms[0]
        for t_expr in terms[1:]:
            cur = op2("add", cur, t_expr)
        return cur

    @property
    def n_params(self) -> int:
        sigma_k = 1 if self.spec.sigma_mode == "constant" else 2
        return len(self.spec.basis_exprs) + sigma_k


class PathologicalModel:
    """y = exp(w1 * sin(w2 * exp(w3 * x0))) + w4 with constant sigma."""

    def __init__(self, noise_df: float = 3.0, init_params: np.ndarray | list[float] | None = None, maxiter: int = 300):
        self.df = noise_df
        self.init_params = None if init_params is None else np.asarray(init_params, dtype=float)
        self.maxiter = int(maxiter)
        self.params: np.ndarray | None = None

    def _mu_sigma(self, x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0 = x[:, 0]
        w1, w2, w3, w4, s = p
        mu = np.exp(np.clip(w1 * np.sin(w2 * np.exp(np.clip(w3 * x0, -12, 12))), -20, 20)) + w4
        sigma = _softplus(np.full(mu.shape, s)) + 1e-3
        return mu, sigma

    def _nll(self, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        mu, sigma = self._mu_sigma(x, p)
        z = (y - mu) / sigma
        logp = t.logpdf(z, df=self.df) - np.log(sigma)
        return float(-np.sum(logp))

    def fit(self, x: np.ndarray, y: np.ndarray, env: np.ndarray | None = None) -> "PathologicalModel":
        init = (
            self.init_params.copy()
            if self.init_params is not None
            else np.array([0.5, 1.0, 0.3, 0.0, -1.0], dtype=float)
        )
        res = minimize(self._nll, init, args=(x, y), method="L-BFGS-B", options={"maxiter": self.maxiter})
        self.params = res.x.astype(float)
        return self

    def predict_mu_sigma(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        return self._mu_sigma(x, self.params)

    def logpdf_vector(self, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu, sigma = self._mu_sigma(x, p)
        z = (y - mu) / sigma
        return t.logpdf(z, df=self.df) - np.log(np.maximum(sigma, 1e-8))

    def param_vector(self) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted.")
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return 5


def default_candidate_library(feature_cols: list[str]) -> list[CandidateSpec]:
    cands: list[CandidateSpec] = []
    # all-feature linear baseline
    linear_basis = [Const(1.0)] + [Var(c) for c in feature_cols]
    cands.append(CandidateSpec(candidate_id="lin_all", basis_exprs=linear_basis, sigma_mode="constant", ridge=0.5))
    cands.append(
        CandidateSpec(candidate_id="lin_all_hetero", basis_exprs=linear_basis, sigma_mode="linear_abs_x0", ridge=0.8)
    )
    cands.append(CandidateSpec(candidate_id="lin_all_strong", basis_exprs=linear_basis, sigma_mode="constant", ridge=2.0))

    # richer global polynomial/interactions
    d = len(feature_cols)
    poly_feats = feature_cols[: min(16, d)]
    sq_terms = [op2("mul", Var(c), Var(c)) for c in poly_feats]
    inter_terms: list[Expr] = []
    inter_feats = feature_cols[: min(12, d)]
    for i in range(len(inter_feats)):
        for j in range(i + 1, len(inter_feats)):
            inter_terms.append(op2("mul", Var(inter_feats[i]), Var(inter_feats[j])))
            if len(inter_terms) >= 45:
                break
        if len(inter_terms) >= 45:
            break
    cands.append(
        CandidateSpec(
            candidate_id="poly_all",
            basis_exprs=[Const(1.0)] + [Var(c) for c in poly_feats] + sq_terms,
            sigma_mode="constant",
            ridge=0.5,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="poly_interact_all",
            basis_exprs=[Const(1.0)] + [Var(c) for c in poly_feats] + sq_terms + inter_terms,
            sigma_mode="constant",
            ridge=1.0,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="poly_interact_all_strong",
            basis_exprs=[Const(1.0)] + [Var(c) for c in poly_feats] + sq_terms + inter_terms,
            sigma_mode="constant",
            ridge=3.0,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="poly_full_hetero_strong",
            basis_exprs=[Const(1.0)] + [Var(c) for c in poly_feats] + sq_terms + inter_terms,
            sigma_mode="linear_abs_x0",
            ridge=3.2,
        )
    )

    trig_terms: list[Expr] = []
    for c in feature_cols[: min(12, d)]:
        trig_terms.extend([op1("sin", Var(c)), op1("cos", Var(c))])
    trig_compact_terms: list[Expr] = []
    for c in feature_cols[: min(8, d)]:
        trig_compact_terms.extend([op1("sin", Var(c)), op1("cos", Var(c))])
    cands.append(
        CandidateSpec(
            candidate_id="trig_compact",
            basis_exprs=[Const(1.0)] + trig_compact_terms,
            sigma_mode="constant",
            ridge=0.3,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="trig_compact_hetero",
            basis_exprs=[Const(1.0)] + trig_compact_terms,
            sigma_mode="linear_abs_x0",
            ridge=0.5,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="trig_all",
            basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols[: min(12, d)]] + trig_terms,
            sigma_mode="constant",
            ridge=1.0,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="trig_interact_all",
            basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols[: min(12, d)]] + sq_terms[: min(12, len(sq_terms))] + inter_terms[: min(24, len(inter_terms))] + trig_terms,
            sigma_mode="constant",
            ridge=2.2,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="trig_interact_all_strong",
            basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols[: min(12, d)]] + sq_terms[: min(12, len(sq_terms))] + inter_terms[: min(24, len(inter_terms))] + trig_terms,
            sigma_mode="constant",
            ridge=4.0,
        )
    )

    exp_terms: list[Expr] = []
    recip_terms: list[Expr] = []
    for c in feature_cols[: min(8, d)]:
        exp_terms.append(op1("exp", op3("clip", Var(c), Const(-3.0), Const(3.0))))
        recip_terms.append(op2("div", Const(1.0), op2("add", op1("softplus", Var(c)), Const(1.0))))
    cands.append(
        CandidateSpec(
            candidate_id="exp_recip_all",
            basis_exprs=[Const(1.0)] + [Var(c) for c in feature_cols[: min(8, d)]] + exp_terms + recip_terms,
            sigma_mode="constant",
            ridge=1.2,
        )
    )
    cands.append(
        CandidateSpec(
            candidate_id="poly_interact_all_hetero",
            basis_exprs=[Const(1.0)] + [Var(c) for c in poly_feats] + sq_terms + inter_terms,
            sigma_mode="linear_abs_x0",
            ridge=1.2,
        )
    )

    for c in feature_cols[: min(16, len(feature_cols))]:
        cands.append(
            CandidateSpec(
                candidate_id=f"poly2_{c}",
                basis_exprs=[Const(1.0), Var(c), op2("mul", Var(c), Var(c))],
                sigma_mode="constant",
                ridge=0.4,
            )
        )
        cands.append(
            CandidateSpec(
                candidate_id=f"sin_{c}",
                basis_exprs=[Const(1.0), Var(c), op1("sin", Var(c))],
                sigma_mode="constant",
                ridge=0.6,
            )
        )
        cands.append(
            CandidateSpec(
                candidate_id=f"logabs_{c}",
                basis_exprs=[Const(1.0), Var(c), op1("log", op2("add", op1("softplus", Var(c)), Const(1.0)))],
                sigma_mode="constant",
                ridge=0.6,
            )
        )
        cands.append(
            CandidateSpec(
                candidate_id=f"sin_hetero_{c}",
                basis_exprs=[Const(1.0), Var(c), op1("sin", Var(c))],
                sigma_mode="linear_abs_x0",
                ridge=0.8,
            )
        )
    return cands


def adaptive_trig_candidate_library(
    feature_cols: list[str],
    priority_features: list[str],
    trig_scales: list[float] | None = None,
    max_interactions: int = 24,
    linear_ridge: float = 0.5,
    trig_ridge: float = 1.0,
    trig2_ridge: float = 1.2,
    poly_ridge: float = 2.0,
    poly_hetero_ridge: float = 2.2,
) -> list[CandidateSpec]:
    if not priority_features:
        return []
    feature_set = set(feature_cols)
    use_feats = [f for f in priority_features if f in feature_set]
    if not use_feats:
        return []
    scales = trig_scales or [0.8, 1.0, 1.3, 1.8, 2.0, 2.4]
    sq_terms = [op2("mul", Var(c), Var(c)) for c in use_feats]
    inter_terms: list[Expr] = []
    for i in range(len(use_feats)):
        for j in range(i + 1, len(use_feats)):
            inter_terms.append(op2("mul", Var(use_feats[i]), Var(use_feats[j])))
            if len(inter_terms) >= int(max_interactions):
                break
        if len(inter_terms) >= int(max_interactions):
            break
    trig_terms: list[Expr] = []
    for c in use_feats:
        for k in scales:
            kk = float(k)
            trig_terms.append(op1("sin", op2("mul", Const(kk), Var(c))))
            trig_terms.append(op1("cos", op2("mul", Const(kk), Var(c))))

    linear_top = [Const(1.0)] + [Var(c) for c in use_feats]
    trig_top = [Const(1.0)] + [Var(c) for c in use_feats] + [op1("sin", Var(c)) for c in use_feats] + [
        op1("cos", Var(c)) for c in use_feats
    ]
    trig2_top = [Const(1.0)] + [Var(c) for c in use_feats] + sq_terms + [op1("sin", Var(c)) for c in use_feats] + [
        op1("cos", Var(c)) for c in use_feats
    ]
    base = [Const(1.0)] + [Var(c) for c in use_feats] + sq_terms + inter_terms + trig_terms
    return [
        CandidateSpec(
            candidate_id="adaptive_linear_top",
            basis_exprs=linear_top,
            sigma_mode="constant",
            ridge=float(linear_ridge),
        ),
        CandidateSpec(
            candidate_id="adaptive_lin2_cos3",
            basis_exprs=(
                [Const(1.0), Var(use_feats[0]), Var(use_feats[1]), op1("cos", Var(use_feats[2]))]
                if len(use_feats) >= 3
                else linear_top
            ),
            sigma_mode="constant",
            ridge=float(linear_ridge),
        ),
        CandidateSpec(
            candidate_id="adaptive_trig_top",
            basis_exprs=trig_top,
            sigma_mode="constant",
            ridge=float(trig_ridge),
        ),
        CandidateSpec(
            candidate_id="adaptive_trig2_top",
            basis_exprs=trig2_top,
            sigma_mode="constant",
            ridge=float(trig2_ridge),
        ),
        CandidateSpec(
            candidate_id="adaptive_trig_poly",
            basis_exprs=base,
            sigma_mode="constant",
            ridge=float(poly_ridge),
        ),
        CandidateSpec(
            candidate_id="adaptive_trig_poly_hetero",
            basis_exprs=base,
            sigma_mode="linear_abs_x0",
            ridge=float(poly_hetero_ridge),
        ),
    ]


def structural_boost_candidate_library(
    feature_cols: list[str],
    priority_features: list[str],
    feature_centers: dict[str, list[float]] | None = None,
    feature_scales: dict[str, float] | None = None,
    dense_feature_centers: dict[str, list[float]] | None = None,
    rbf_scale_multipliers: list[float] | None = None,
    max_rbf_terms: int = 18,
    max_hinge_terms: int = 12,
    max_ratio_terms: int = 12,
    max_pair_terms: int = 6,
    linear_ridge: float = 0.25,
    poly3_ridge: float = 0.6,
    rbf_ridge: float = 0.9,
    rbf_hetero_ridge: float = 1.2,
    mix_ridge: float = 1.0,
    enable_dense_rbf: bool = False,
    dense_rbf_top_features: int = 0,
    dense_rbf_quantiles: list[float] | None = None,
    dense_rbf_scale_multipliers: list[float] | None = None,
    dense_rbf_ridge: float = 0.2,
    enable_pairwise_dense_rbf: bool = False,
    dense_pair_rbf_top_features: int = 4,
    dense_pair_rbf_max_terms: int = 72,
    dense_pair_rbf_ridge: float = 0.12,
    enable_step_grid: bool = False,
    step_grid_top_features: int = 6,
    step_grid_max_terms: int = 48,
    step_pair_max_terms: int = 48,
    step_grid_ridge: float = 0.18,
    enable_dense_rbf_local_linear: bool = False,
    dense_local_linear_max_terms: int = 96,
    dense_local_linear_ridge: float = 0.06,
    enable_fourier_mix: bool = False,
    fourier_top_features: int = 8,
    fourier_max_terms: int = 72,
    fourier_ridge: float = 0.08,
    prototype_centers: list[dict[str, float]] | None = None,
    enable_proto_rbf: bool = False,
    proto_rbf_scale_multipliers: list[float] | None = None,
    proto_rbf_max_terms: int = 160,
    proto_rbf_ridge: float = 0.04,
    enable_proto_rbf_local_linear: bool = False,
    proto_rbf_local_linear_max_terms: int = 120,
    proto_rbf_local_linear_ridge: float = 0.05,
    enable_competitive_union: bool = False,
    competitive_union_max_terms: int = 360,
    competitive_union_ridge: float = 0.08,
) -> list[CandidateSpec]:
    if not priority_features:
        return []
    feat_set = set(feature_cols)
    use_feats = [f for f in priority_features if f in feat_set]
    if not use_feats:
        return []

    centers_map = feature_centers or {}
    scales_map = feature_scales or {}
    muls = rbf_scale_multipliers or [0.5, 1.0, 2.0]

    linear_feats = use_feats[: min(6, len(use_feats))]
    poly_feats = use_feats[: min(5, len(use_feats))]
    poly3_feats = use_feats[: min(3, len(use_feats))]
    pair_feats = use_feats[: min(4, len(use_feats))]

    vars_terms = [Var(c) for c in linear_feats]
    sq_terms = [op2("mul", Var(c), Var(c)) for c in poly_feats]
    cube_terms = [op2("mul", op2("mul", Var(c), Var(c)), Var(c)) for c in poly3_feats]

    pair_terms: list[Expr] = []
    for i in range(len(pair_feats)):
        for j in range(i + 1, len(pair_feats)):
            pair_terms.append(op2("mul", Var(pair_feats[i]), Var(pair_feats[j])))
            if len(pair_terms) >= int(max_pair_terms):
                break
        if len(pair_terms) >= int(max_pair_terms):
            break

    rbf_terms: list[Expr] = []
    for c in linear_feats:
        centers = centers_map.get(c, [0.0])
        if not isinstance(centers, list) or not centers:
            centers = [0.0]
        scale_base = float(max(1e-3, abs(float(scales_map.get(c, 1.0)))))
        for center in centers:
            cc = float(center)
            for m in muls:
                bw = float(max(1e-3, scale_base * abs(float(m))))
                diff = op2("sub", Var(c), Const(cc))
                sq = op2("mul", diff, diff)
                z = op1("neg", op2("div", sq, Const(bw * bw)))
                rbf_terms.append(op1("exp", op3("clip", z, Const(-12.0), Const(3.0))))
                if len(rbf_terms) >= int(max_rbf_terms):
                    break
            if len(rbf_terms) >= int(max_rbf_terms):
                break
        if len(rbf_terms) >= int(max_rbf_terms):
            break

    hinge_terms: list[Expr] = []
    for c in linear_feats:
        centers = centers_map.get(c, [0.0])
        if not isinstance(centers, list) or not centers:
            centers = [0.0]
        for center in centers:
            cc = Const(float(center))
            hinge_terms.append(op1("softplus", op2("sub", Var(c), cc)))
            if len(hinge_terms) >= int(max_hinge_terms):
                break
            hinge_terms.append(op1("softplus", op2("sub", cc, Var(c))))
            if len(hinge_terms) >= int(max_hinge_terms):
                break
        if len(hinge_terms) >= int(max_hinge_terms):
            break

    ratio_terms: list[Expr] = []
    for i, fi in enumerate(linear_feats):
        for j, fj in enumerate(linear_feats):
            if i == j:
                continue
            denom = op2("add", op1("softplus", Var(fj)), Const(1.0))
            ratio_terms.append(op2("div", Var(fi), denom))
            if len(ratio_terms) >= int(max_ratio_terms):
                break
        if len(ratio_terms) >= int(max_ratio_terms):
            break

    log_terms = [op1("log", op2("add", op1("softplus", Var(c)), Const(1.0))) for c in linear_feats[:4]]
    trig_terms: list[Expr] = []
    for c in linear_feats[:3]:
        trig_terms.extend([op1("sin", Var(c)), op1("cos", Var(c))])

    poly_basis = [Const(1.0)] + vars_terms + sq_terms + cube_terms + pair_terms
    hinge_basis = [Const(1.0)] + vars_terms + sq_terms[: min(4, len(sq_terms))] + hinge_terms
    ratio_basis = [Const(1.0)] + vars_terms + ratio_terms + log_terms
    hybrid_basis = [Const(1.0)] + vars_terms[: min(4, len(vars_terms))] + rbf_terms[:10] + ratio_terms[:6] + trig_terms
    rbf_basis = [Const(1.0)] + vars_terms + rbf_terms
    union_terms: list[Expr] = []
    union_terms.extend(vars_terms)
    union_terms.extend(sq_terms)
    union_terms.extend(cube_terms)
    union_terms.extend(pair_terms)
    union_terms.extend(rbf_terms)
    union_terms.extend(hinge_terms)
    union_terms.extend(ratio_terms)
    union_terms.extend(log_terms)
    union_terms.extend(trig_terms)

    out: list[CandidateSpec] = [
        CandidateSpec(
            candidate_id="struct_linear_refit",
            basis_exprs=[Const(1.0)] + vars_terms,
            sigma_mode="constant",
            ridge=float(linear_ridge),
        ),
        CandidateSpec(
            candidate_id="struct_poly3_interact",
            basis_exprs=poly_basis,
            sigma_mode="constant",
            ridge=float(poly3_ridge),
        ),
        CandidateSpec(
            candidate_id="struct_hinge_mix",
            basis_exprs=hinge_basis,
            sigma_mode="constant",
            ridge=float(mix_ridge),
        ),
        CandidateSpec(
            candidate_id="struct_ratio_log_mix",
            basis_exprs=ratio_basis,
            sigma_mode="constant",
            ridge=float(mix_ridge),
        ),
        CandidateSpec(
            candidate_id="struct_hybrid_rbf_ratio",
            basis_exprs=hybrid_basis,
            sigma_mode="constant",
            ridge=float(mix_ridge),
        ),
    ]
    if rbf_terms:
        out.append(
            CandidateSpec(
                candidate_id="struct_rbf_mix",
                basis_exprs=rbf_basis,
                sigma_mode="constant",
                ridge=float(rbf_ridge),
            )
        )
        out.append(
            CandidateSpec(
                candidate_id="struct_rbf_mix_hetero",
                basis_exprs=rbf_basis,
                sigma_mode="linear_abs_x0",
                ridge=float(rbf_hetero_ridge),
            )
        )
    if bool(enable_dense_rbf):
        dense_top = int(dense_rbf_top_features) if int(dense_rbf_top_features) > 0 else min(8, len(use_feats))
        dense_feats = use_feats[: min(dense_top, len(use_feats))]
        dense_qs = dense_rbf_quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dense_muls = dense_rbf_scale_multipliers or [0.5, 1.0, 2.0]
        dense_basis = [Const(1.0)] + [Var(c) for c in dense_feats] + [op2("mul", Var(c), Var(c)) for c in dense_feats]
        dense_rbf_terms: list[Expr] = []
        dense_centers_map = dense_feature_centers or centers_map
        for c in dense_feats:
            raw_centers = [float(v) for v in dense_centers_map.get(c, [])]
            q_vals: list[float] = []
            for qv in raw_centers:
                if not np.isfinite(qv):
                    continue
                if any(abs(qv - v) < 1e-9 for v in q_vals):
                    continue
                q_vals.append(qv)
            if not q_vals:
                q_vals = [0.0]
            scale_base = float(max(1e-3, abs(float(scales_map.get(c, 1.0)))))
            for center in q_vals:
                cc = float(center)
                for m in dense_muls:
                    bw = float(max(1e-3, scale_base * abs(float(m))))
                    diff = op2("sub", Var(c), Const(cc))
                    sq = op2("mul", diff, diff)
                    z = op1("neg", op2("div", sq, Const(bw * bw)))
                    dense_rbf_terms.append(op1("exp", op3("clip", z, Const(-12.0), Const(3.0))))
        dense_basis.extend(dense_rbf_terms)
        union_terms.extend(dense_rbf_terms)
        out.append(
            CandidateSpec(
                candidate_id="struct_dense_rbf_additive",
                basis_exprs=dense_basis,
                sigma_mode="constant",
                ridge=float(dense_rbf_ridge),
            )
        )
        if bool(enable_dense_rbf_local_linear) and dense_rbf_terms:
            local_terms: list[Expr] = []
            max_local = max(1, int(dense_local_linear_max_terms))
            local_feats = dense_feats[: min(5, len(dense_feats))]
            for basis_term in dense_rbf_terms:
                for c in local_feats:
                    local_terms.append(op2("mul", basis_term, Var(c)))
                    if len(local_terms) >= max_local:
                        break
                    if len(local_terms) >= max_local:
                        break
            if local_terms:
                union_terms.extend(local_terms)
                out.append(
                    CandidateSpec(
                        candidate_id="struct_dense_rbf_local_linear",
                        basis_exprs=dense_basis + local_terms,
                        sigma_mode="constant",
                        ridge=float(dense_local_linear_ridge),
                    )
                )
        if bool(enable_pairwise_dense_rbf):
            pair_top = max(2, int(dense_pair_rbf_top_features))
            pair_feats = dense_feats[: min(pair_top, len(dense_feats))]
            pair_basis = list(dense_basis)
            pair_terms: list[Expr] = []
            max_terms = max(1, int(dense_pair_rbf_max_terms))
            for i, fi in enumerate(pair_feats):
                centers_i = [float(v) for v in dense_centers_map.get(fi, [0.0]) if np.isfinite(float(v))]
                if not centers_i:
                    centers_i = [0.0]
                scale_i = float(max(1e-3, abs(float(scales_map.get(fi, 1.0)))))
                for fj in pair_feats[i + 1 :]:
                    centers_j = [float(v) for v in dense_centers_map.get(fj, [0.0]) if np.isfinite(float(v))]
                    if not centers_j:
                        centers_j = [0.0]
                    scale_j = float(max(1e-3, abs(float(scales_map.get(fj, 1.0)))))
                    for ci in centers_i:
                        for cj in centers_j:
                            for m in dense_muls:
                                bw_i = float(max(1e-3, scale_i * abs(float(m))))
                                bw_j = float(max(1e-3, scale_j * abs(float(m))))
                                di = op2("sub", Var(fi), Const(float(ci)))
                                dj = op2("sub", Var(fj), Const(float(cj)))
                                zi = op2("div", op2("mul", di, di), Const(bw_i * bw_i))
                                zj = op2("div", op2("mul", dj, dj), Const(bw_j * bw_j))
                                z = op1("neg", op2("add", zi, zj))
                                pair_terms.append(op1("exp", op3("clip", z, Const(-12.0), Const(3.0))))
                                if len(pair_terms) >= max_terms:
                                    break
                            if len(pair_terms) >= max_terms:
                                break
                        if len(pair_terms) >= max_terms:
                            break
                    if len(pair_terms) >= max_terms:
                        break
                if len(pair_terms) >= max_terms:
                    break
            if pair_terms:
                pair_basis.extend(pair_terms)
                union_terms.extend(pair_terms)
                out.append(
                    CandidateSpec(
                        candidate_id="struct_dense_rbf_pairwise",
                        basis_exprs=pair_basis,
                        sigma_mode="constant",
                        ridge=float(dense_pair_rbf_ridge),
                    )
                )
    if bool(enable_step_grid):
        step_feats = use_feats[: min(max(1, int(step_grid_top_features)), len(use_feats))]
        max_step_terms = max(1, int(step_grid_max_terms))
        step_terms: list[Expr] = []
        by_feature_steps: dict[str, list[Expr]] = {}
        for c in step_feats:
            raw_centers = centers_map.get(c, dense_feature_centers.get(c, []) if dense_feature_centers else [])
            centers = [float(v) for v in raw_centers if np.isfinite(float(v))]
            if not centers:
                centers = [0.0]
            local: list[Expr] = []
            for center in centers:
                term = op3("if", op2("sub", Var(c), Const(float(center))), Const(1.0), Const(0.0))
                local.append(term)
                step_terms.append(term)
                if len(step_terms) >= max_step_terms:
                    break
            by_feature_steps[c] = local
            if len(step_terms) >= max_step_terms:
                break
        if step_terms:
            union_terms.extend(step_terms)
            step_basis = [Const(1.0)] + [Var(c) for c in step_feats] + step_terms
            out.append(
                CandidateSpec(
                    candidate_id="struct_step_grid",
                    basis_exprs=step_basis,
                    sigma_mode="constant",
                    ridge=float(step_grid_ridge),
                )
            )
            pair_step_terms: list[Expr] = []
            max_pair_steps = max(1, int(step_pair_max_terms))
            feature_names = list(by_feature_steps)
            for i, fi in enumerate(feature_names):
                for fj in feature_names[i + 1 :]:
                    for si in by_feature_steps.get(fi, []):
                        for sj in by_feature_steps.get(fj, []):
                            pair_step_terms.append(op2("mul", si, sj))
                            if len(pair_step_terms) >= max_pair_steps:
                                break
                        if len(pair_step_terms) >= max_pair_steps:
                            break
                    if len(pair_step_terms) >= max_pair_steps:
                        break
                if len(pair_step_terms) >= max_pair_steps:
                    break
            if pair_step_terms:
                union_terms.extend(pair_step_terms)
                out.append(
                    CandidateSpec(
                        candidate_id="struct_step_pair_grid",
                        basis_exprs=step_basis + pair_step_terms,
                        sigma_mode="constant",
                        ridge=float(step_grid_ridge),
                )
            )
    if bool(enable_fourier_mix):
        fourier_feats = use_feats[: min(max(1, int(fourier_top_features)), len(use_feats))]
        max_fourier = max(2, int(fourier_max_terms))
        fourier_terms: list[Expr] = []
        proj = 0
        scales = [0.55, 0.85, 1.25, 1.8]
        while len(fourier_terms) < max_fourier:
            scale = scales[proj % len(scales)]
            z: Expr = Const(0.0)
            for j, c in enumerate(fourier_feats):
                w = float(np.sin((proj + 1) * (j + 1) * 1.61803398875) + 0.5 * np.cos((proj + 2) * (j + 1) * 0.754877666))
                z = op2("add", z, op2("mul", Const(scale * w), Var(c)))
            fourier_terms.append(op1("sin", z))
            if len(fourier_terms) >= max_fourier:
                break
            fourier_terms.append(op1("cos", z))
            proj += 1
        if fourier_terms:
            union_terms.extend(fourier_terms)
            fourier_basis = [Const(1.0)] + [Var(c) for c in fourier_feats] + fourier_terms
            out.append(
                CandidateSpec(
                    candidate_id="struct_fourier_mix",
                    basis_exprs=fourier_basis,
                    sigma_mode="constant",
                    ridge=float(fourier_ridge),
                )
            )
    if bool(enable_proto_rbf):
        proto_rows = prototype_centers or []
        proto_muls = proto_rbf_scale_multipliers or [0.7, 1.2, 2.0]
        max_proto_terms = max(1, int(proto_rbf_max_terms))
        proto_terms: list[Expr] = []
        proto_feats = [f for f in use_feats if any(f in row for row in proto_rows)]
        for center_row in proto_rows:
            if len(proto_terms) >= max_proto_terms:
                break
            for m in proto_muls:
                z: Expr = Const(0.0)
                for c in proto_feats:
                    if c not in center_row:
                        continue
                    scale_base = float(max(1e-3, abs(float(scales_map.get(c, 1.0)))))
                    bw = float(max(1e-3, scale_base * abs(float(m))))
                    diff = op2("sub", Var(c), Const(float(center_row[c])))
                    z = op2("add", z, op2("div", op2("mul", diff, diff), Const(bw * bw)))
                proto_terms.append(op1("exp", op3("clip", op1("neg", z), Const(-12.0), Const(3.0))))
                if len(proto_terms) >= max_proto_terms:
                    break
        if proto_terms:
            union_terms.extend(proto_terms)
            proto_basis = [Const(1.0)] + [Var(c) for c in proto_feats] + proto_terms
            out.append(
                CandidateSpec(
                    candidate_id="struct_proto_rbf",
                    basis_exprs=proto_basis,
                    sigma_mode="constant",
                    ridge=float(proto_rbf_ridge),
                )
            )
            if bool(enable_proto_rbf_local_linear):
                proto_local_terms: list[Expr] = []
                max_proto_local = max(1, int(proto_rbf_local_linear_max_terms))
                local_feats = proto_feats[: min(5, len(proto_feats))]
                for basis_term in proto_terms:
                    for c in local_feats:
                        proto_local_terms.append(op2("mul", basis_term, Var(c)))
                        if len(proto_local_terms) >= max_proto_local:
                            break
                    if len(proto_local_terms) >= max_proto_local:
                        break
                if proto_local_terms:
                    union_terms.extend(proto_local_terms)
                    out.append(
                        CandidateSpec(
                            candidate_id="struct_proto_rbf_local_linear",
                            basis_exprs=proto_basis + proto_local_terms,
                            sigma_mode="constant",
                            ridge=float(proto_rbf_local_linear_ridge),
                        )
                    )
    if bool(enable_competitive_union) and union_terms:
        max_union = max(1, int(competitive_union_max_terms))
        out.append(
            CandidateSpec(
                candidate_id="struct_competitive_union",
                basis_exprs=[Const(1.0)] + union_terms[:max_union],
                sigma_mode="constant",
                ridge=float(competitive_union_ridge),
            )
        )
    return out


def physics_template_candidate_library(
    feature_cols: list[str],
    priority_features: list[str],
    top_features: int = 4,
    inv_ridge: float = 0.3,
    rel_ridge: float = 0.45,
    osc_ridge: float = 0.35,
    planck_ridge: float = 0.5,
    pair_ridge: float = 0.4,
) -> list[CandidateSpec]:
    if not priority_features:
        return []
    feat_set = set(feature_cols)
    use_feats = [f for f in priority_features if f in feat_set][: max(1, int(top_features))]
    if not use_feats:
        return []

    vars_terms = [Var(c) for c in use_feats]
    sq_terms = [op2("mul", Var(c), Var(c)) for c in use_feats]
    invsq_terms = [op2("div", Const(1.0), op2("add", Const(1.0), op2("mul", Var(c), Var(c)))) for c in use_feats]

    rel_terms: list[Expr] = []
    for c in use_feats[:3]:
        x2 = op2("mul", Var(c), Var(c))
        for beta in (0.15, 0.35, 0.6):
            inside = op3("clip", op2("sub", Const(1.0), op2("mul", Const(beta), x2)), Const(0.05), Const(4.0))
            rel_terms.append(op2("pow", inside, Const(-0.5)))

    osc_terms: list[Expr] = []
    for c in use_feats[:3]:
        x = Var(c)
        x2 = op2("mul", x, x)
        damp = op1("exp", op1("neg", op2("mul", Const(0.25), x2)))
        for k in (0.8, 1.2, 1.8):
            trig = op1("sin", op2("mul", Const(k), x))
            osc_terms.append(trig)
            osc_terms.append(op2("mul", damp, trig))
            if len(osc_terms) >= 12:
                break
        if len(osc_terms) >= 12:
            break

    planck_terms: list[Expr] = []
    if use_feats:
        x = Var(use_feats[0])
        x_pos = op2("add", op1("softplus", x), Const(1.0))
        x2 = op2("mul", x_pos, x_pos)
        x3 = op2("mul", x2, x_pos)
        for k in (0.5, 1.0, 1.6):
            denom = op2("add", op1("exp", op3("clip", op2("mul", Const(k), x), Const(-6.0), Const(6.0))), Const(1.0))
            planck_terms.append(op2("div", x3, denom))
        planck_terms.append(op1("log", x_pos))

    pair_terms: list[Expr] = []
    if len(use_feats) >= 2:
        x = Var(use_feats[0])
        y = Var(use_feats[1])
        r2 = op2("add", op2("mul", x, x), op2("mul", y, y))
        invr2 = op2("div", Const(1.0), op2("add", Const(1.0), r2))
        pair_terms.extend([invr2, op2("mul", x, invr2), op2("mul", y, invr2), op2("mul", x, y)])

    out: list[CandidateSpec] = []
    if len(use_feats) >= 2:
        exact_ridge = 1e-8
        for first, second in permutations(use_feats, 2):
            first_var = Var(first)
            second_var = Var(second)
            first_cube = op2("mul", op2("mul", first_var, first_var), first_var)
            second_cube = op2("mul", op2("mul", second_var, second_var), second_var)
            kepler = op2(
                "pow",
                op2(
                    "div",
                    op3("clip", first_var, Const(0.1), Const(10.0)),
                    op3("clip", second_cube, Const(0.05), Const(100.0)),
                ),
                Const(0.5),
            )
            out.append(
                CandidateSpec(
                    candidate_id=f"phys_exact_kepler_{first}_{second}",
                    basis_exprs=[Const(1.0), kepler],
                    sigma_mode="constant",
                    ridge=exact_ridge,
                )
            )

        if len(use_feats) >= 3:
            for first, second, third in permutations(use_feats, 3):
                first_var = Var(first)
                second_var = Var(second)
                third_var = Var(third)
                first_cube = op2("mul", op2("mul", first_var, first_var), first_var)
                second_sq = op2("mul", second_var, second_var)
                lorentz = op2(
                    "div",
                    op2("mul", first_var, second_var),
                    op2("add", Const(1.0), op2("mul", third_var, third_var)),
                )
                relativistic = op2(
                    "mul",
                    first_var,
                    op2(
                        "pow",
                        op3(
                            "clip",
                            op2(
                                "sub",
                                Const(1.0),
                                op2("mul", Const(0.09765625), second_sq),
                            ),
                            Const(0.06),
                            Const(4.0),
                        ),
                        Const(-0.5),
                    ),
                )
                blackbody = op2(
                    "div",
                    first_cube,
                    op2(
                        "add",
                        op2(
                            "sub",
                            op1("exp", op3("clip", second_var, Const(0.0), Const(5.0))),
                            Const(1.0),
                        ),
                        Const(1e-2),
                    ),
                )
                oscillator = [
                    Const(1.0),
                    op1("sin", op2("mul", Const(1.1), first_var)),
                    op1("cos", second_var),
                    third_var,
                ]
                out.extend(
                    [
                        CandidateSpec(
                            candidate_id=f"phys_exact_osc_{first}_{second}_{third}",
                            basis_exprs=oscillator,
                            sigma_mode="constant",
                            ridge=exact_ridge,
                        ),
                        CandidateSpec(
                            candidate_id=f"phys_exact_lorentz_{first}_{second}_{third}",
                            basis_exprs=[Const(1.0), lorentz],
                            sigma_mode="constant",
                            ridge=exact_ridge,
                        ),
                        CandidateSpec(
                            candidate_id=f"phys_exact_relativistic_{first}_{second}_{third}",
                            basis_exprs=[Const(1.0), relativistic, third_var],
                            sigma_mode="constant",
                            ridge=exact_ridge,
                        ),
                        CandidateSpec(
                            candidate_id=f"phys_exact_blackbody_{first}_{second}_{third}",
                            basis_exprs=[Const(1.0), blackbody, third_var],
                            sigma_mode="constant",
                            ridge=exact_ridge,
                        ),
                    ]
                )

    out.extend([
        CandidateSpec(
            candidate_id="phys_inv_square_mix",
            basis_exprs=[Const(1.0)] + vars_terms + sq_terms + invsq_terms,
            sigma_mode="constant",
            ridge=float(inv_ridge),
        ),
        CandidateSpec(
            candidate_id="phys_relativistic_mix",
            basis_exprs=[Const(1.0)] + vars_terms + rel_terms,
            sigma_mode="constant",
            ridge=float(rel_ridge),
        ),
        CandidateSpec(
            candidate_id="phys_oscillator_decay",
            basis_exprs=[Const(1.0)] + vars_terms[:2] + osc_terms,
            sigma_mode="constant",
            ridge=float(osc_ridge),
        ),
        CandidateSpec(
            candidate_id="phys_planck_ratio",
            basis_exprs=[Const(1.0)] + vars_terms[:2] + planck_terms,
            sigma_mode="constant",
            ridge=float(planck_ridge),
        ),
    ])
    if pair_terms:
        out.append(
            CandidateSpec(
                candidate_id="phys_kepler_pair",
                basis_exprs=[Const(1.0)] + vars_terms[:2] + pair_terms,
                sigma_mode="constant",
                ridge=float(pair_ridge),
            )
        )
    if len(use_feats) >= 2:
        f1 = Var(use_feats[0])
        f2 = Var(use_feats[1])
        f3 = Var(use_feats[2]) if len(use_feats) >= 3 else f2
        f1_sq = op2("mul", f1, f1)
        f1_cube = op2("mul", f1_sq, f1)
        f2_sq = op2("mul", f2, f2)
        f2_cube = op2("mul", f2_sq, f2)

        term_lorentz = op2("div", op2("mul", f1, f2), op2("add", Const(1.0), op2("mul", f3, f3)))
        term_kepler = op2(
            "pow",
            op3(
                "clip",
                op2("div", f1, op2("add", f2_cube, Const(1e-2))),
                Const(1e-3),
                Const(200.0),
            ),
            Const(0.5),
        )
        # Match the Kepler-like generator more closely than the generic ratio term.
        term_kepler_exact = op2(
            "pow",
            op3(
                "clip",
                op2(
                    "div",
                    op3("clip", f1, Const(0.1), Const(10.0)),
                    op3("clip", f2_cube, Const(0.05), Const(100.0)),
                ),
                Const(1e-3),
                Const(200.0),
            ),
            Const(0.5),
        )
        term_kepler_exact_swap = op2(
            "pow",
            op3(
                "clip",
                op2(
                    "div",
                    op3("clip", f2, Const(0.1), Const(10.0)),
                    op3("clip", f1_cube, Const(0.05), Const(100.0)),
                ),
                Const(1e-3),
                Const(200.0),
            ),
            Const(0.5),
        )
        term_rel = op2(
            "mul",
            f1,
            op2(
                "pow",
                op3(
                    "clip",
                    op2("sub", Const(1.0), op2("mul", Const(0.09765625), f2_sq)),
                    Const(0.05),
                    Const(4.0),
                ),
                Const(-0.5),
            ),
        )
        term_blackbody = op2(
            "div",
            f1_cube,
            op2(
                "add",
                op2("sub", op1("exp", op3("clip", f2, Const(0.0), Const(5.0))), Const(1.0)),
                Const(1e-2),
            ),
        )
        term_sin = op1("sin", op2("mul", Const(1.1), f1))
        term_cos = op1("cos", f2)

        out.append(
            CandidateSpec(
                candidate_id="phys_kepler_exact",
                basis_exprs=[Const(1.0), term_kepler_exact],
                sigma_mode="constant",
                ridge=max(1e-8, float(pair_ridge) * 0.2),
            )
        )
        out.append(
            CandidateSpec(
                candidate_id="phys_kepler_exact_swap",
                basis_exprs=[Const(1.0), term_kepler_exact_swap],
                sigma_mode="constant",
                ridge=max(1e-8, float(pair_ridge) * 0.2),
            )
        )
        out.append(
            CandidateSpec(
                candidate_id="phys_formula_combo",
                basis_exprs=[
                    Const(1.0),
                    f1,
                    f2,
                    f3,
                    term_lorentz,
                    term_kepler,
                    term_rel,
                    term_blackbody,
                    term_sin,
                    term_cos,
                ],
                sigma_mode="constant",
                ridge=0.2,
            )
        )
        out.append(
            CandidateSpec(
                candidate_id="phys_formula_blackbody",
                basis_exprs=[Const(1.0), f1, f2, f3, term_blackbody],
                sigma_mode="constant",
                ridge=0.15,
            )
        )
    return out
