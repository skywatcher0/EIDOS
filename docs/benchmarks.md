# Dataset workflows

The dataset runner trains EIDOS from scratch using `configs/benchmark.json`. Its numerical engine is the installed `eidos` package. The frozen configuration also supplies the public estimator defaults.

## Collections

`benchmark` uses 243 PMLB samples: 119 Feynman laws, 62 Friedman tasks, 14 Strogatz systems, five FirstPrinciples/Nikuradse tasks and 43 general regression tasks. `openml` uses fifteen complete numeric tables and ten deterministic partitions each. The source catalogues are in `data/pmlb/catalogue.csv` and `data/openml/catalogue.csv`.

The input sample seed is `42 + (CRC32(dataset_name) & 0x7fffffff) + 7919 * partition`. PMLB samples are already fixed; OpenML tables are subsampled to 300 rows. Five quantile groups of the first raw feature define split strata. The proportions are 40/15/15/15/15, with deterministic integer allocation within each stratum. Discovery-only screening retains at most eight features, and discovery-only standardization precedes response quantization at 0.001.

The dataset runner retains these sampling, quantile tie and metric conventions for repeatability. The general CSV/API interface accepts arbitrary sample counts and uses equal-count strata with stable tie handling. Its default seed is 42 and its metrics use the unrounded response in original units. Consequently a generic CSV fit and a fixed dataset example need not have the same split or score.

## Results

Every collection writes:

| Path | Contents |
|---|---|
| `<collection>/raw.csv` | Every requested task/partition, including explicit failure rows |
| `<collection>/summary.csv` | PMLB family summaries or OpenML dataset summaries |
| `<collection>/artifacts/<dataset>__<partition>/model.json` | Predictor loadable by the CLI and `EIDOSRegressor.load` |
| `.../record.json` | Exact construction record without calibration state |
| `.../preprocessing.json` | Selected features, affine transforms and source-row indices |
| `.../prepared.csv` | Standardized observations and evidence roles |
| `.../test_predictions.csv` | Quantized-target predictions, bounds and raw-unit predictions |
| `.../replacement_check.json` | PMLB record comparison after later-role response replacement |
| `<collection>/reference_differences.csv` | Differences from fixed numerical reference values |
| `verification_report.json`, `summary.json`, `environment.json` | Checks, summaries, package versions and run fingerprint |

`rmse` is measured in discovery-standardized units against quantized targets; `rmse_raw_units` multiplies it by the discovery target scale. It therefore differs slightly from error against unrounded targets. Model-record decoding must preserve predictions exactly within a run. Each PMLB task is also fitted after jointly replacing verification, calibration and test responses; its construction payload must remain identical.

The checked numerical fixtures under `reference/` are never used for fitting or feature selection. Comparisons use `rtol=1e-5`, `atol=1e-8`; PMLB RMSE and interval widths allow `atol=1e-5` (one hundredth of a response bin), with stricter differences still recorded. Candidate identities are compared exactly. Runtime and bytes across different numerical library versions are not promised identical.

## Decoding a dataset model

Use the trained model directly on a numeric table in its original units:

```sh
python -m eidos predict --model results/benchmarks/benchmark/artifacts/feynman_I_6_2a__0/model.json --input data/pmlb/samples/feynman_I_6_2a.csv.gz --output results/feynman_predictions.csv
```

The API accepts the same file with `EIDOSRegressor.load`. Its interval radius follows the dataset workflow's quantized calibration responses. To inspect the lower-level construction record:

```python
import json
import numpy as np
import pandas as pd
from eidos.serialization import unpack

record = json.load(open("results/benchmarks/benchmark/artifacts/feynman_I_6_2a__0/record.json"))
model = unpack(record)
rows = pd.read_csv("results/benchmarks/benchmark/artifacts/feynman_I_6_2a__0/prepared.csv")
prediction, residual_scale = model.predict_mu_sigma(rows[model.feature_cols].to_numpy(float))
center, scale = record["construction_context"]["target_affine"]
prediction_raw_units = prediction * scale + center
```

## Compute and resuming

The full workflow performs 636 fits: 243 PMLB fits, 243 replacement checks and 150 OpenML fits. Each worker uses one numerical-library thread. `--workers` controls process parallelism. The smoke collection performs five fits.

Use a dedicated `--output` directory for each run. `--resume` reuses successful cells only when code, configuration and bundled input fingerprints agree. Reinstall the package after modifying `src/`, or use an editable development installation (`python -m pip install --no-deps -e .`). Running `verify_results.py` reads existing outputs and does not refit models.
