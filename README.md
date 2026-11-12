# EIDOS

EIDOS learns compact regression models from numeric tables. It combines symbolic basis search, quantized coefficients, independent model verification and calibrated prediction intervals. A saved JSON record contains the transformations and model state needed to make predictions on another machine.

Use EIDOS with your own CSV files or explore the bundled scientific and engineering datasets through the same modeling engine.

## Install

Use **64-bit Python 3.12** for the locked environment. Python 3.13 is also supported. CPU execution is the default.

```sh
git clone https://github.com/skywatcher0/EIDOS.git
cd EIDOS
```

Create and activate an environment:

```sh
# Linux / macOS
python3.12 -m venv .venv
. .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install the locked dependencies and the package:

```sh
python -m pip install --only-binary=:all: -r requirements-lock.txt
python -m pip install --no-deps .
python -m eidos --version
```

If activation is unavailable, replace `python` with `.venv/bin/python` on Linux/macOS or `.\.venv\Scripts\python.exe` on Windows. The repository also works when downloaded as a ZIP and extracted.

For a minimal library installation, `python -m pip install .` installs only the numerical dependencies. The scripts that download benchmark sources additionally need `python -m pip install ".[benchmarks]"`.

## Train and predict

The included example has two numeric features and a continuous target:

```sh
python -m eidos fit --input examples/demo.csv --target target --output results/demo
python -m eidos predict --model results/demo/model.json --input examples/demo.csv --output results/demo_predictions.csv
```

Replace the input path and target name to use your own data:

```sh
python -m eidos fit --input measurements.csv --target response --features temperature pressure flow --seed 42 --output results/measurements
python -m eidos predict --model results/measurements/model.json --input new_measurements.csv --output results/new_predictions.csv
```

Inputs must contain at least 80 complete rows, finite numeric features and a varying numeric target. Use `--features` to exclude identifiers or other columns that should not be predictors. Without it, all columns except the target and any declared role/environment columns are features. Encode categories and handle missing values before training. Prediction files need only the selected feature columns; column order does not matter for named CSV inputs.

Each training run creates:

| File | Use |
|---|---|
| `model.json` | Load the fitted predictor and interval calibration without training data |
| `metrics.json` | Test RMSE, MAE, coverage, verification statistic and role counts |
| `test_predictions.csv` | Held-out targets, predictions and interval bounds |
| `roles.csv` | Row assignments for inspecting or repeating the split |
| `formula.json` | Readable expression and the transformations connecting it to input units |

Predictions, RMSE and interval bounds use the original target units. The displayed expression uses standardized coordinates and rounded constants; use `model.json` for exact numerical inference. Existing non-empty output directories are rejected so a new fit does not silently replace a model.

## Python API

```python
import pandas as pd
from eidos import EIDOSRegressor

data = pd.read_csv("examples/demo.csv")
X, y = data[["x", "z"]], data["target"]

model = EIDOSRegressor(random_state=42, max_features=8, alpha=0.1)
model.fit(X, y)
print(model.metrics_)
print(model.formula_)

model.save("results/model.json")
loaded = EIDOSRegressor.load("results/model.json")
predictions = loaded.predict(X.head())
intervals = loaded.predict_interval(X.head())  # columns: lower, upper
```

`predict` also accepts a NumPy matrix in the original training feature order. Named DataFrames may provide just `selected_features_`. See [API and model records](docs/usage.md) for explicit evidence roles, environment labels and model inspection.

## How training uses data

EIDOS assigns rows to five disjoint roles: **40% discovery, 15% diagnosis, 15% verification, 15% calibration and 15% test**. Discovery determines feature selection, transformations and candidate construction. Discovery cross-fitting and diagnosis select a structure, which is refitted using those two roles. Verification evaluates the fixed model; calibration sets the interval radius; test rows measure performance.

The default candidate bank includes polynomial, rational, trigonometric, radial-basis and power-law components, with a sparse combination stage. At most eight input features are retained by default, using discovery-only correlations. Coefficients are quantized on a precision-selected grid and predictions use the decoded values.

`fit` deliberately reserves evidence rows instead of training on every input row. A 90% split-conformal interval (`alpha=0.1`) has its coverage interpretation when calibration and future observations are exchangeable; under distribution shift, inspect observed coverage for your application. The verification statistic is a diagnostic, not an automatic rejection rule. See [modeling workflow](docs/usage.md).

## Explore datasets and check repeatability

The datasets are ready to use after installation; default runs require no further downloads.

```sh
# Short check: two PMLB datasets and one OpenML partition
python run_benchmarks.py --suite smoke --workers 2

# All included regression tasks
python run_benchmarks.py --suite all --workers 4 --output results/benchmarks

# Run either collection independently
python run_benchmarks.py --suite benchmark --workers 4 --output results/pmlb
python run_benchmarks.py --suite openml --workers 4 --output results/openml
```

| Collection | Work performed |
|---|---|
| PMLB | 243 tasks: 200 mechanism/synthetic tasks and 43 general regression tasks; one fresh fit and one evidence-replacement check per task |
| OpenML | 15 scientific and engineering datasets, each with ten independent partitions: 150 fresh fits |

Each PMLB input contains the fixed sample of up to 300 rows used by that example. OpenML includes cleaned complete tables and deterministically selects up to 300 rows per partition. Feature screening and scaling are fitted on discovery rows. The dataset runner preserves its fixed sampling and split conventions; see [benchmark guide](docs/benchmarks.md).

Runs write per-task scores, decoded models, row assignments and test predictions, followed by summaries and a `verification_report.json`. Reference values are used only for checks after fitting. The process exits unsuccessfully if a task fails or a check disagrees. Timings depend on the computer and are not pass/fail criteria. Start with two workers on a laptop; the full collection takes longer than the smoke check and runs entirely on CPU.

Dataset runs save the same loadable model format as custom training. For example:

```sh
python -m eidos predict --model results/benchmarks/benchmark/artifacts/feynman_I_6_2a__0/model.json --input data/pmlb/samples/feynman_I_6_2a.csv.gz --output results/feynman_predictions.csv
```

To resume an interrupted run or recheck its output:

```sh
python run_benchmarks.py --suite all --workers 4 --output results/benchmarks --resume
python verify_results.py results/benchmarks
```

A shortcut creates `.venv`, installs the package and runs the small collection:

```sh
python bootstrap.py --suite smoke --workers 2
```

## Data sources

[Data guide](data/README.md) lists included inputs, processing and source attribution. [Full PMLB download links](data/pmlb/DOWNLOADS.md) and [OpenML metadata](data/openml/sources.json) accompany the data. The 232 MB uncompressed `1191_BNG_pbc` source is provided through a [download and reconstruction workflow](data/LARGE_DATA.md); its fixed 300-row input is already bundled.

```sh
python verify_data.py
python prepare_data.py --track pmlb --datasets 1191_BNG_pbc --download-missing --output rebuilt_data
python prepare_data.py --track openml --datasets yacht_hydrodynamics --from-source --output rebuilt_data
```

## Development and containers

```sh
python -m unittest discover -s tests -v
python -m pip wheel --no-deps . --wheel-dir dist
docker build -t eidos .
docker run --rm -v "${PWD}/container_results:/results" eidos
```

The Docker image runs the small dataset collection by default. Override the command with `run_benchmarks.py --suite all --workers 4 --output /results` for the full collection. The automated workflow checks installation, model persistence, the CLI and a small dataset run on Linux, Windows and macOS.

Software and data licensing information is in [LICENSE_NOTICE.txt](LICENSE_NOTICE.txt).
