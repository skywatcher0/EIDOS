# Using EIDOS

## Input and preprocessing

Use a DataFrame with unique string column names or a two-dimensional numeric NumPy array. Supply a one-dimensional continuous response. Missing values, infinity and text-valued features are rejected. Rows are never silently dropped by the estimator. Constant discovery features are excluded; if no feature or response varies on discovery rows, fitting reports an error.

Discovery rows determine means, scales and, when needed, the `max_features` most correlated columns. The default is eight. Targets are standardized and rounded to response bins of width 0.001 for model selection. The public estimator computes its final test errors and calibrates intervals against the original unrounded response, in original units.

Keep identifiers out of `X`. If a preprocessing step learns from responses, fit it using the discovery rows as well. The model preserves feature transformations so new rows should be passed in their original units.

## Explicit environments and roles

Without environment labels, the estimator creates five equal-count strata from the first feature, breaking ties by input row order. These are covariate partitions, not inferred causal environments. With supplied labels, splitting occurs independently within each group. Every group needs at least five rows for automatic allocation, and discovery must contain at least twelve rows overall.

```python
model.fit(X, y, environments=batch_labels)
```

For an existing split, pass one of `disc`, `diag`, `audit`, `cal`, `test` for every row. All roles must be present. `audit` is the verification role.

```python
model.fit(X, y, environments=batch_labels, roles=role_labels)
```

The command-line equivalent accepts label columns that are excluded from predictors:

```sh
python -m eidos fit --input measurements.csv --target response --environment batch --role role --output results/grouped
```

A fit reserves rows for model checking and calibration. It does not later refit on the test set. Changing verification, calibration or test responses with fixed role assignments leaves the constructed predictor unchanged; calibration changes may alter the interval radius, and evaluation changes may alter the reported metrics.

## Estimator interface

| Member | Meaning |
|---|---|
| `EIDOSRegressor(random_state=42, max_features=8, alpha=0.1)` | Deterministic seed, feature limit and interval error level |
| `fit(X, y, environments=None, roles=None, dataset_id="custom")` | Construct, evaluate and calibrate; returns the estimator |
| `predict(X)` | One prediction per row in original target units |
| `predict_interval(X)` | Two columns: lower and upper bounds |
| `save(path)` / `EIDOSRegressor.load(path)` | Write/read a self-contained JSON predictor |
| `selected_features_` | Original names retained for prediction |
| `role_assignments_` | Role for each input row; available after fitting |
| `metrics_` / `test_predictions_` | Held-out evaluation; available after fitting |
| `formula_` | Readable expression in standardized coordinates |
| `record_` / `model_sha256_` | Exact construction record and checksum |

Keep `dataset_id` fixed when comparing reruns: it participates in deterministic internal seeds. For arrays, preserve the original feature order. For DataFrames, columns are selected by name. Predictions can be generated from a loaded model in any working directory; the source training data and checkout are unnecessary.

## Saved model

`model.json` stores expression trees, binary64 hexadecimal parameters, basis normalization, coefficient grid, selected features and discovery transformations. It uses a data-only decoder, not pickle or executable expressions. JSON file size is a transport cost, not the MDL objective.

The construction record is independent of calibration. The outer model file separately stores the interval radius and coverage setting. A checksum protects the construction record against accidental changes. Human-readable expressions round constants; use the saved record when exact numerical predictions matter.

The displayed expression returns a standardized response. For each retained feature use `x_i = (original_feature - center_i) / scale_i`; then convert the expression output to `prediction = target_center + target_scale * expression`. These affine values are in `record_["construction_context"]` and `formula.json`. `predict` performs these steps automatically.

## Intervals and diagnostics

The estimator uses the finite-sample split-conformal rank `ceil((n_cal + 1) * (1 - alpha))` of absolute calibration residuals. A calibration set too small for the requested coverage gives infinite bounds, which are preserved through save/load. Choose `alpha` before inspecting test outcomes.

The verification p-value checks residual behavior across environment labels for the fixed model. It does not certify scientific causality. Intervals describe prediction uncertainty under exchangeability; shifted operating conditions require application-level evaluation.
