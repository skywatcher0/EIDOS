# EIDOS: Representation-Complete and Evidence-Isolated Admission of Machine-Discovered Equations into Expert Systems

This repository provides a self-contained reference implementation of the EIDOS pipeline described in the ESWA manuscript *EIDOS: Representation-Complete and Evidence-Isolated Admission of Machine-Discovered Equations into Expert Systems*.

This release expands the original three-case demonstration to the main EIDOS-only experiments in `EIDOS_ESWA_v12_reviewed.zip`, using the compact reproduction package. Competing methods and ablation experiments are outside this release. [中文运行说明](README_zh.md).

## Method overview

When a symbolic-regression (SR) pipeline delivers an equation to a knowledge base, two questions must have clear answers before the entry can be trusted: (1) *what exactly was selected* — whether the finite expression, its coefficients, precision, and decoding rules are sufficient to reconstruct the object that was actually scored; and (2) *which evidence selected it* — whether the observations used for final acceptance were genuinely independent of all discovery and selection decisions. Failure on the first question is called **representation leakage** (the stored entry, the scored object, and a reconstructed object may be three different things); failure on the second is called **evidence leakage** (holdout figures lose their intended interpretation when selection quietly consumes them).

EIDOS closes both channels simultaneously through two complementary constructions.

**Representation channel.** A versioned context `(C₀, C)` serves as a frozen task ontology — a typed vocabulary, admissibility rules, coding conventions, and deterministic settings. Every candidate receives prefix-decodable structure and parameter serialisations, plus a normalised probability mass over discretised responses. The ideal conditional joint codelength satisfies a Kraft inequality (Theorem 1 of the paper); its ceiling yields a complete-record prefix code with less than one bit of overhead. Every scored and reported model record is therefore exactly reconstructible from its transmitted description and the declared context.

**Evidence channel.** Observations are partitioned into five disjoint roles — discovery, diagnosis, verification, calibration, and test — with a strict data-flow contract. Discovery freezes the context and instantiates the finite candidate bank; diagnosis scores and selects; the selected structure is refitted on discovery-plus-diagnosis data. Verification, calibration, and test act only on the fixed reported model and never feed back into selection or refit. The canonical model payload is therefore invariant to later-role replacement: under independent stateless reruns, byte-level payload comparison detects every violation that changes the payload, with zero false alarms under compliance (Propositions 2–4 of the paper).

The executable experiments follow their frozen study protocols. The main PMLB and OpenML tracks use the final EIDOS implementation. Reproducing the reported recovery curve and interval figure requires the archived study implementations: recovery cross-fitting uses discovery-plus-diagnosis rows, and the interval study retains its earlier core. These differ from subsequent protocol changes described by the previous three-case demonstration. The source snapshots and exact differences are documented in [PROTOCOL_NOTES.md](docs/PROTOCOL_NOTES.md).

## How it works

The reproduction covers prediction, equation recovery and interval evaluation, exercising the EIDOS pipeline end to end:

1. **Data preparation.** The exact experimental inputs for 243 PMLB tasks and 15 OpenML datasets are included. Controlled recovery observations are generated locally from the declared equations, noise levels and seeds; the interval study uses its bundled input tables.

2. **Candidate-bank construction.** The main benchmark uses its frozen EIDOS configuration. The controlled recovery study constructs the original task-independent bank of 240 anonymised candidates from five physics-inspired operator families, before scoring begins.

3. **Five-role allocation.** Observations are assigned to disjoint discovery, diagnosis, verification, calibration and test roles. The recovery study uses the original 40% / 15% / 15% / 15% / 15% allocation; each other study retains its recorded configuration.

4. **Candidate scoring and selection.** Every requested cell performs fresh candidate fitting and selection. In the archived recovery study, initial fits use discovery rows, cross-fit RMSE uses discovery-plus-diagnosis rows, and diagnosis RMSE and description length serve as ordered tie-breakers. Verification, calibration and test rows stay outside construction.

5. **Post-selection refit.** The selected structure is refitted on pooled discovery-plus-diagnosis data, re-quantised onto the declared parameter grid, and canonically serialised.

6. **Evaluation.** Predictions are evaluated on the test role. Recovery semantic NRMSE uses noiseless generating values and their test-set standard deviation. The interval study reports coverage and width for the frozen interval procedures.

7. **Multi-seed verification.** The recovery and interval studies use five registered seeds; OpenML uses ten partitions per dataset. Every PMLB task is fitted again after later-role replacement to check payload invariance. Reference comparisons occur only after computation.

## Quick start

Use 64-bit Python 3.12. From a fresh checkout:

```sh
# Linux / macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-lock.txt
.venv/bin/python reproduce.py --suite all --workers 4
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py --suite all --workers 4
```

Or use the one-command helpers, which create `.venv`, install the locked dependencies and run the requested suite:

```sh
sh run_reproduction.sh --suite all --workers 4          # Linux / macOS
.\run_reproduction.ps1 --suite all --workers 4          # Windows PowerShell
```

Use `--suite smoke` for a short installation check, or select `benchmark`, `openml`, `recovery` or `conformal` individually. Internet access is needed to install dependencies; all default experimental inputs are bundled.

## Expected output

A complete `--suite all` run produces the following EIDOS results:

| Suite | Fresh computation | Paper result covered |
|---|---|---|
| `benchmark` | 243 PMLB tasks, plus 243 later-role replacement fits | Predictive performance, record size, quantisation and replacement conformance |
| `openml` | 15 datasets × 10 partitions = 150 fits | Public OpenML predictive results |
| `recovery` | 5 generators × 8 noise levels × 5 seeds = 200 fits | Recovery curves and illustrative equation records |
| `conformal` | 5 generators × 2 arms × 5 seeds = 50 fits; 125 interval rows | Pooled, Mondrian and auxiliary proxy-trend interval results on EIDOS predictions |

The original three illustrative cases remain included. At noise σ = 0.80, all five seeds recover the generating structure for each case; the first-seed rows produce these semantic NRMSE values:

| Case | Selected candidate | Semantic NRMSE |
|---|---|---:|
| Oscillator | `cand_0454` | `0.083221` |
| Lorentz | `cand_0446` | `0.038477` |
| Blackbody proxy | `cand_0596` | `0.005398` |

`reproduce.py` exits with a non-zero status if a requested experiment fails or a result check disagrees with the reference. The complete compact-package validation passed all 29 checks. Candidate identities and recovery flags are checked exactly. Numerical comparisons use `rtol=1e-5, atol=1e-8`, with a portable `atol=1e-5` for PMLB RMSE and interval width; stricter comparisons and differences remain in the report. Runtime is not a pass/fail criterion.

This release reproduces the included EIDOS results, not comparator-dependent ranks, win counts, significance tests or every table in the paper.

## Output files

A successful full run writes the following files under `results/`; a smoke run defaults to `results_smoke/`:

| File | Content |
|---|---|
| `benchmark/raw.csv`, `openml/raw.csv` | Every requested task or partition, including any explicit failure rows |
| `recovery/raw.csv`, `conformal/raw.csv` | Recovery fits and interval-result rows |
| `recovery/formula_examples.csv` | The original three displayed equation records with formulas and NRMSE |
| `recovery/recovery.png`, `recovery/recovery.pdf` | Recovery plots from fresh results |
| `conformal/coverage.png`, `conformal/coverage.pdf` | Coverage plots from fresh results |
| `<suite>/summary.csv`, `summary.json` | Per-study and combined summaries |
| `benchmark/artifacts/`, `openml/artifacts/`, `recovery/artifacts/` | Model records, data roles, predictions and study-specific audit files |
| `verification_report.json`, `<suite>/reference_differences.csv` | Field-level reproduction checks and differences |
| `environment.json` | Python version, platform, package versions, worker settings and input/code fingerprint |

Use `--output` to retain multiple runs:

```sh
.venv/bin/python reproduce.py --suite all --output results/run_02
```

Add `--resume` to reuse successful cells only when their method, configuration and input fingerprints match. Saved benchmark/OpenML model records can also be decoded for prediction without refitting through `predict.py`.

## Tests

Core invariants — role isolation, response quantisation, normalised probabilities and exact model-record decoding — are checked by:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python verify_manifest.py
```

The experiment runner checks the requested task grids and numerical agreement after fitting. To check an existing result directory again:

```sh
.venv/bin/python verify_results.py results
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python`.

## Data and compute

The compact distribution contains the exact experimental samples for all 243 PMLB tasks, the cleaned numeric tables for all 15 OpenML datasets, the five interval-study input tables, and audit copies of the 200 generated recovery datasets. Recovery fitting regenerates its observations from the declared equations and seeds.

Complete PMLB source tables are supplied as [download links](data/pmlb/DOWNLOADS.md) and a reconstruction script instead of being bundled. This includes the source table above 100 MB. Sources, checksums and processing metadata are recorded in `data/pmlb/manifest.json` and `data/openml/`; see [LARGE_DATA.md](data/LARGE_DATA.md).

To download, reconstruct and compare the inputs with the bundled experimental data:

```sh
.venv/bin/python prepare_data.py --track pmlb --download-missing --output rebuilt_data
.venv/bin/python prepare_data.py --track openml --from-source --output rebuilt_data
.venv/bin/python prepare_data.py --track recovery --output rebuilt_data
```

The PMLB command downloads all 243 source tables by default. Add `--datasets 706_sleuth_case1202,feynman_I_6_2a` for a small reconstruction check. Normal experiment runs need none of these source downloads.

The run is CPU-only, using one numerical-library thread per worker. Four workers and 8 GB RAM are a practical starting point. The independent compact-package full run took 214.48 seconds with eight workers on the recorded Windows machine; runtime varies by processor. [COMPACT_VALIDATION.md](validation/COMPACT_VALIDATION.md) records that run, and [VALIDATION.md](validation/VALIDATION.md) records the original Windows/Linux validation and numerical differences.

`validation/` contains example verification outputs and is never read for fitting. The paper reference in `reference/` is used only after fresh computation for result verification — it is not an input to data generation, selection or fitting.

## Provenance

The included results correspond to `EIDOS_ESWA_v12_reviewed.zip`, SHA-256 `478ab27a8e6ab4869cc5bae03d718f8688b4112d6f680a1cd113c363c2e73016`. The manuscript source is identified in `paper/source.json`.

This release uses `EIDOS_ESWA_v12_Reproduction_Compact.zip` (26,349,805 bytes), SHA-256 `e00cc947ada9ef31bf39c2f44b186b5fc0c6fbd9a7d6ff0e90fb948398eb7519`. The scientific code, configurations and data are preserved from that package. `MANIFEST.sha256` records the distributed repository files.

[PROVENANCE.md](docs/PROVENANCE.md) maps the inputs and configurations to their sources. [PROTOCOL_NOTES.md](docs/PROTOCOL_NOTES.md) records the recovery and interval study differences from the later implementation; reproducing those archived numerical results does not resolve that protocol discrepancy. Dataset attribution and the existing licensing boundary are retained in `LICENSE_NOTICE.txt`.
