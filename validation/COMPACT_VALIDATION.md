# Compact-package validation — 2026-11-05

The compact ZIP was extracted into a separate directory with no full PMLB
source tables. All experiments were run again from that extracted package,
without result-cache reuse, using the clean Python 3.12 environment and the
locked dependencies from the original release validation.

| Fresh experiment | Completed |
|---|---:|
| PMLB prediction tasks | 243/243 |
| PMLB later-role replacement checks | 243/243 |
| OpenML dataset/partition fits | 150/150 |
| Recovery fits | 200/200 |
| Conformal fits | 50/50 |
| Interval-result rows | 125/125 |
| Verification checks | 29/29 |

Full-run wall time: **214.48 seconds**, using eight
CPU worker processes, excluding dependency installation and source downloads.
The numerical tolerances and study-specific protocols are unchanged.

`compact_windows_full/` contains the fresh raw results, summaries, environment
record and verification report. These are verification evidence, not training
inputs. The original Windows/Linux full-release validation is also retained
in this archive; the new compact run was performed on Windows.

All **346** files listed in `METHOD_AND_INPUTS.sha256` match the full
release exactly. No scientific implementation, experiment, partition or
experimental input was removed. The 242 full PMLB source tables previously
included in the archive were removed from the distribution; all 243 full
source tables now have external download links, with `raw_bundled=false`
in the data manifest. All 243 exact experimental samples remain bundled.

The full-source download and reconstruction path was separately exercised
from the extracted compact package for `706_sleuth_case1202` and
`feynman_I_6_2a`. Both were freshly downloaded and their reconstructed samples
matched the bundled data within the existing `1e-12` numerical tolerance.
See `data_preparation/compact_source_downloads.json` for the actual checks.
The default experiment run requires none of these downloads.

The final ZIP is checked file by file against `MANIFEST.sha256`, and its CRC
is verified. The companion `.zip.sha256` file records the archive digest.

精简版保留全部实验与实际使用的数据。已从独立解压目录重新运行完整实验，
全部 29 项结果检查通过；下载重建脚本也已实际测试。
安装依赖后，默认复现无需下载完整 PMLB 原始表。
