# Provenance and frozen experimental scope

The controlling source is the user-designated `EIDOS_ESWA_v12_reviewed.zip`,
not an older release with a larger legacy version number. Its extracted
scientific text and source digest are supplied in `paper/`.

| Package component | Original project source | Adaptation |
|---|---|---|
| `src/eidos/` | `src/eidos/`; engine also matches the archived v6 experiment-code snapshot used by the v12 line | Byte-for-byte copy; original source hashes in `core_source_provenance.json` |
| `configs/benchmark.json` | `scripts/run_srbench100_extension.py::_fast_eidos_cfg(strict_v34,99)` plus the final `v8` specification in `scripts/v6/variants.py` | Resolve to a plain JSON configuration; no development variants included as experiments |
| `repro/data.py` | `scripts/v6/task_cache.py` and `scripts/v6/run_real_track.py` | Preserve seed, sample, split, screening, quantisation and affine-transformation logic; use round-trip CSV instead of pickle |
| `experiments/recovery.py` | R2 final delivery source snapshot, `scripts/run_eswa_e5_recovery.py` | Remove old CLI/output paths; preserve the actual construction-role cross-fit protocol that generated Figure 3; add observation-only score ledger |
| `experiments/conformal.py` and `experiments/interval_core/` | R2 final delivery source snapshot, interval driver and `src/eidos/` | Resolve local input paths; isolate archived core imports under a separate namespace; preserve the numerical algorithm |
| `reference/benchmark.csv` | `outputs/v6_improvement/variants/formal_trace.csv`, final `v8` rows only | 243 EIDOS rows; remove variant labels and development traces |
| `reference/openml.csv` | `outputs/v6_improvement/real_track/results.csv`, `eidos:v8` only | 150 EIDOS rows, last record per dataset/partition |
| `reference/recovery_noise_sweep.csv` | `outputs/eidos_method_only_improvement_20260823/formal/recovery_noise_sweep.csv` | Preserve 200 archived rows |
| `reference/conformal_two_arm.csv` | Same formal-output directory, `conformal_two_arm.csv` | Preserve 125 interval rows |
| `data/pmlb/samples/` | `outputs/v6_improvement/raw_task_cache/*.pkl` | Lossless numeric CSV conversion; only rows used by the paper |
| `data/pmlb/manifest.json`, `data/pmlb/DOWNLOADS.md` | Original source metadata and `outputs/revision_experiments_20260831/pmlb_cache/` checksums | Full tables are external in the compact edition; retain all URLs and the reconstruction script |
| `data/openml/raw/` | `outputs/v6_improvement/real_track/raw/*.pkl` | Lossless numeric CSV conversion; full cleaned datasets |
| `data/conformal/` | Five `data/processed_v34_components/feynman_like_*` inputs | Preserve data and metadata |
| `data/recovery/` | Registered generator code, eight noise levels, five seeds | Deterministically regenerated CSVs |

Historical directory names above are provenance labels only. They are not
paths that users must create. The code reads no file outside this archive.

The final main benchmark uses seed `42 + (crc32(dataset_name) & 0x7fffffff)`.
OpenML partition seeds add `7919 * partition` for partitions 0–9. The fitting
seed on OpenML is reduced modulo `2**31-1`, matching the original runner.
PMLB raw sampling uses at most 300 rows; OpenML repeats 300-row subsampling
for each partition. Both use first-feature proxy strata and five role ratios
`0.40/0.15/0.15/0.15/0.15`. `audit` is the implementation's name for the
paper's verification role.

Recovery uses generators `oscillator, lorentz, relativistic, kepler, blackbody`,
noise levels `0.01,0.05,0.10,0.20,0.40,0.80,1.60,3.20`, and seeds
`20260225,20260226,20260227,20260228,20260229`. Its discovery-only initial candidate
fits, discovery+diagnosis cross-fitting, diagnosis-RMSE and codelength
tie-breaks, and discovery+diagnosis final refit are retained from the source
that generated the reported recovery sweep. That snapshot has no later-added
diagnosis hard gate or selected-structure ridge grid. These differences from
the current checkout are explicitly audited in `PROTOCOL_NOTES.md`.
Truth labels in the candidate
mapping are used only for post-selection scoring, never to select a candidate.

Conformal evaluates the five bundled 1,900-row generator datasets with the
same five seeds. Exchangeable and highest-stratum-held-out arms are both
included. The proxy-trend envelope is an auxiliary interval construction on
the EIDOS predictions, not a competing regression method or an ablation.

Reference metrics serve as verification evidence, never as fit inputs.
If a platform produces a numerical or structural difference, the driver
keeps the newly computed result and emits an explicit failed comparison.
