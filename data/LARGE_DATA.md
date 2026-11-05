# Full source data downloads (compact edition)

All 243 complete PMLB source tables are external in this compact edition.
The exact experimental samples are bundled and support every default run
offline after dependency installation.

- [All 243 direct source-download links](pmlb/DOWNLOADS.md)
- [Machine-readable URLs, sizes and archived checksums](pmlb/manifest.json)

Reconstruct every full source table and verify its experimental sample:

```sh
python prepare_data.py --track pmlb --download-missing --output rebuilt_data
```

Downloads are cached under `rebuilt_data/downloads/`. Reconstructing the full
sources is optional, requires network access and uses substantially more disk
space than running the supplied experimental samples.

## The source above 100 MB

The threshold is 100,000,000 bytes of the decompressed raw table.
`1191_BNG_pbc` is the only full PMLB table in this scope above the threshold
(approximately 232 MB; exact sizes are in `pmlb/manifest.json`).

- [Dataset source and metadata](https://github.com/EpistasisLab/pmlb/tree/master/datasets/1191_BNG_pbc)
- [Direct gzip download](https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/1191_BNG_pbc/1191_BNG_pbc.tsv.gz)

From the package directory, with its Python environment:

```sh
python prepare_data.py --track pmlb --datasets 1191_BNG_pbc --download-missing --output rebuilt_data
```

The script downloads the table, coerces features and response to numeric,
drops incomplete rows, selects up to 300 rows without replacement using the
registered dataset-name seed, writes portable CSV and original row indices,
and compares the numerical result with the exact bundled experimental sample.
The default experiment does not need this download because the 300-row
sample actually used in the paper is already included.

The remaining 242 full PMLB raw tables are also available by download in the
compact edition. A regenerated upstream gzip need not have the same byte hash as
the archived CSV serialization, so the script also verifies the numerical
data used by the experiment.
