# Delivery validation — 2026-11-05

This is the archived validation record of the original full-data release.
The compact edition retains the same scientific code and experimental inputs;
its additional validation is recorded in `COMPACT_VALIDATION.md`. References
below to 242 bundled full raw tables describe the earlier release, not the
compact archive.

## Completed fresh computation

| Check | Windows | Linux |
|---|---:|---:|
| PMLB tasks | 243/243 | 243/243 |
| PMLB later-role replacement invariance | 243/243 | 243/243 |
| OpenML dataset/partition fits | 150/150 | 150/150 |
| Recovery generator/noise/seed fits | 200/200 | 200/200 |
| Conformal generator/arm/seed fits | 50/50 | 50/50 |
| Interval-result rows | 125/125 | 125/125 |
| Final verification checks | 29/29 | 29/29 |
| Full-run wall time, 8 workers | 301.11 s | 168.44 s |

Both runs started without result caches. Each evaluates 643 primary experiment
cells plus 243 replacement reruns. Numerical libraries use one thread per
worker. Timing excludes dependency installation and data reconstruction and
is a measurement on the validation machine, not a promised runtime on other
hardware.

- Windows: clean Python 3.12.12 environment with the pinned numerical packages.
- Linux: Python 3.12.13, x86-64 WSL2 Linux, glibc 2.27, independently installed
  Linux wheels; copied package under a Linux home directory and invoked from
  outside the package. No original EIDOS checkout was on its Python import path.
- A second standalone Windows copy successfully ran the actual `bootstrap.py`
  entry point: it created its own `.venv`, installed all 19 pinned dependencies
  with standard pip and completed the smoke suite in 13.05 s after installation.
- Five contract tests passed: role partitioning, quantiser boundaries, finite-bin
  probability normalisation, lossless model reconstruction and schema rejection.
- The saved-model prediction CLI was exercised successfully on a 93-row input.
- Recovery and coverage plots were visually inspected.
- macOS and the Dockerfile were not executed in this validation.

`windows_full/`, `linux_full/`, and `standalone_bootstrap_smoke/` contain the
actual raw result tables, summaries, software manifests and verification reports.
They are never read by the fitting code. Large per-cell artifacts are generated
by the runner rather than duplicated here.

## Numerical agreement

All Windows metrics pass the strict comparison (`rtol=1e-5, atol=1e-8`).
On Linux all candidate identities, recovery flags, coverage values, verification
p-values and decoder checks agree. One PMLB task has an RMSE difference of
`4.114527176989069e-7` and an interval-width difference of
`2.0095715847073814e-6`. These pass the documented portable allowance of 1%
of the PMLB response-quantisation bin (`0.001 * 0.01 = 1e-5`). The strict
match counts and the two numerical differences remain in the Linux report
and `benchmark/reference_differences.csv`.

The portable threshold was introduced after observing this cross-platform
rounding difference. Both platforms' already computed raw outputs were then
reverified, without replacing values or refitting. It applies only to PMLB
RMSE and interval width; categorical/structural checks remain exact. The
substantive protocol discrepancies in `docs/PROTOCOL_NOTES.md` fail by much
larger margins and were resolved using the archived study source, not by
changing tolerances.

The final driver also sorts fingerprint paths explicitly for identical
Windows/Linux ordering. The full-run environment records retain their
original pre-packaging fingerprints. `METHOD_AND_INPUTS.sha256` identifies
the final numerical method files and inputs; scientific code and inputs were
unchanged by the final reporting/fingerprint-order adjustments.

## Main regenerated quantities

- Exact recovery for noise at or below 0.80: **150/150**.
- At noise 3.20, recovered counts out of five: Blackbody **5**, Lorentz **3**,
  Relativistic **2**, Kepler **1**, Oscillator **0**.
- Exchangeable coverage: pooled **0.9054035**, Mondrian **0.9146667**.
- Shifted coverage: pooled **0.7281053**, nearest-stratum Mondrian **0.8407368**,
  proxy-trend envelope **0.9421053**.
- Main benchmark median basis count: **17**.
- Replacement-invariant construction payloads: **243/243**.

## Data reconstruction

All **243** PMLB samples were reconstructed from the full raw sources.
For 242 bundled raw tables, the numerical samples matched bit for bit.
The omitted large `1191_BNG_pbc` source was downloaded using the provided
URL and script; its reconstructed sample matched at `rtol=atol=1e-12`.
The exact original 300-row sample remains bundled for the default run.

The OpenML source-processing path was tested from a fresh download of dataset
43918 (`energy_efficiency_heating`): 768 rows, removal of alternate target
`Y2`, and agreement with the bundled cleaned table. Metadata, licences and
source URLs were retrieved for all 15 OpenML datasets. Full source re-download
of all 15 OpenML datasets was not required or performed; their bundled tables
were used and fully evaluated in both complete runs.

All 200 recovery CSVs were generated from the provided equations and seeds.
Preparation reports are included in `data_preparation/`.

## Protocol audit

The current checkout alone did not reproduce the full recovery/interval
figures. The initial differing outputs are preserved in
`current_checkout_protocol_audit/`. The delivered defaults use the actual
archived EIDOS study implementations, with separate namespaces and source
hashes. This is an explicit archival reproduction, not a claim that the
manuscript and later revised code already agree on those two protocols.
