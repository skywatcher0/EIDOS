# EIDOS: Representation-Complete and Evidence-Isolated Admission of Machine-Discovered Equations into Expert Systems

This repository provides a self-contained reference implementation of the EIDOS pipeline described in the ESWA manuscript *EIDOS: Representation-Complete and Evidence-Isolated Admission of Machine-Discovered Equations into Expert Systems*.

## Method overview

When a symbolic-regression (SR) pipeline delivers an equation to a knowledge base, two questions must have clear answers before the entry can be trusted: (1) *what exactly was selected* — whether the finite expression, its coefficients, precision, and decoding rules are sufficient to reconstruct the object that was actually scored; and (2) *which evidence selected it* — whether the observations used for final acceptance were genuinely independent of all discovery and selection decisions. Failure on the first question is called **representation leakage** (the stored entry, the scored object, and a reconstructed object may be three different things); failure on the second is called **evidence leakage** (holdout figures lose their intended interpretation when selection quietly consumes them).

EIDOS closes both channels simultaneously through two complementary constructions.

**Representation channel.** A versioned context `(C₀, C)` serves as a frozen task ontology — a typed vocabulary, admissibility rules, coding conventions, and deterministic settings. Every candidate receives prefix-decodable structure and parameter serialisations, plus a normalised probability mass over discretised responses. The ideal conditional joint codelength satisfies a Kraft inequality (Theorem 1 of the paper); its ceiling yields a complete-record prefix code with less than one bit of overhead. Every scored and reported model record is therefore exactly reconstructible from its transmitted description and the declared context.

**Evidence channel.** Observations are partitioned into five disjoint roles — discovery, diagnosis, verification, calibration, and test — with a strict data-flow contract. Discovery freezes the context and instantiates the finite candidate bank; diagnosis scores and selects; the selected structure is refitted on discovery-plus-diagnosis data. Verification, calibration, and test act only on the fixed reported model and never feed back into selection or refit. The canonical model payload is therefore invariant to later-role replacement: under independent stateless reruns, byte-level payload comparison detects every violation that changes the payload, with zero false alarms under compliance (Propositions 2–4 of the paper).

The pipeline selects structure by three-fold discovery cross-fit RMSE, with diagnosis RMSE and description length as ordered tie-breakers. A residual-invariance gate screens candidates via a kernel two-sample statistic on cross-fitted residuals across proxy strata. After selection, only the winning structure's ridge penalty is tuned over a fixed multiplier grid, and the model is refitted on pooled discovery-plus-diagnosis data.

## How it works

The demonstration reconstructs three representative EIDOS equation records (oscillator, Lorentz, and blackbody-proxy generators), exercising the full pipeline end to end:

1. **Data generation.** Synthetic observations are generated locally from declared mechanisms and registered seeds, with additive Gaussian noise (σ = 0.80). No external dataset or network download is required.

2. **Candidate-bank construction.** A task-independent bank of 240 anonymised candidates is built from five physics-inspired operator families. The bank is deterministic and frozen before any scoring begins.

3. **Five-role allocation.** Observations are stratified by proxy environment and split into discovery (40%), diagnosis (15%), audit (15%), calibration (15%), and test (15%).

4. **Candidate scoring and selection.** Each candidate is fitted on discovery data and scored via three-fold cross-fit RMSE. Diagnosis RMSE and the EIDOS description length serve as ordered tie-breakers.

5. **Post-selection refit.** The selected structure is refitted on pooled discovery-plus-diagnosis data, re-quantised onto the declared parameter grid, and canonically serialised.

6. **Semantic evaluation.** The refitted model is evaluated on the test role using noiseless generating values. Semantic NRMSE quantifies how closely the stored expression tracks the true mechanism.

7. **Multi-seed verification.** Five registered seeds are evaluated. The five-seed recovery gate checks that every seed recovers the generating structure, ensuring the result is not seed-dependent.

## Quick start

Python 3.12 is recommended. From a fresh checkout:

```sh
# Linux / macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py
```

Or use the one-command helpers:

```sh
sh run_reproduction.sh          # Linux / macOS
.\run_reproduction.ps1          # Windows PowerShell
```

## Expected output

A successful run recovers the generating structure in all five seeds for each of the three cases. The first-seed rows produce these semantic NRMSE values:

| Case | Selected candidate | Semantic NRMSE |
|---|---|---:|
| Oscillator | `cand_0454` | `0.083221` |
| Lorentz | `cand_0446` | `0.038477` |
| Blackbody proxy | `cand_0596` | `0.005398` |

`reproduce.py` exits with a non-zero status if any candidate identifier, displayed formula, semantic NRMSE, or five-seed recovery gate disagrees with the paper reference.

## Output files

A successful run writes the following files under `results/`:

| File | Content |
|---|---|
| `table5_eidos.csv` | Three displayed EIDOS records with formulas and NRMSE |
| `table5_eidos_all_seeds.csv` | All 15 case × seed fits used by the recovery gate |
| `table5_five_seed_gate.csv` | Per-case `5/5` recovery summary |
| `candidate_bank.csv` | Regenerated 240-entry bank with canonical basis expressions |
| `verification_report.json` | Field-level reproduction status against the paper reference |
| `environment.json` | Python version, platform, package versions, seeds, and noise metadata |

Use `--output-dir` to retain multiple runs:

```sh
.venv/bin/python reproduce.py --output-dir results/run_02
```

## Tests

Core invariants — bank size, role allocation, first-seed formula identity, and numerical agreement — are checked by:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Data and compute

All observations are generated locally from the equations and registered seeds in `table5_experiment.py`. No external dataset or network download is required. The run is CPU-only and deterministic under the pinned environment. Runtime varies by processor; the scientific checks exclude elapsed time.

`reference_run/` records a clean Python 3.12 execution made from the locked dependencies when this package was assembled. It is an example output and is never read by the runner. The compact paper reference in `reference/` is used only after fresh computation for result verification — it is not an input to data generation, selection, or fitting.

## Provenance

The displayed rows correspond to the ESWA v12 manuscript source package dated 2026-08-28, SHA-256 `B0A90B3B01DA48A7119EE5308A40CA2C6091652F7483D4B69D101870BDC0A31C`. `MANIFEST.sha256` records the files in this reproduction package.
