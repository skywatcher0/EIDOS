# Large upstream data

`1191_BNG_pbc` has a 232,149,593-byte uncompressed source table, above the 100 MB threshold. Its 300-row modeling input is already included as `pmlb/samples/1191_BNG_pbc.csv.gz`.

- [Source and metadata](https://github.com/EpistasisLab/pmlb/tree/master/datasets/1191_BNG_pbc)
- [Download the compressed source](https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/1191_BNG_pbc/1191_BNG_pbc.tsv.gz)
- [All source links](pmlb/DOWNLOADS.md)

Download and process it with:

```sh
python prepare_data.py --track pmlb --datasets 1191_BNG_pbc --download-missing --output rebuilt_data
```

The script caches the download, selects complete numeric rows, applies the fixed dataset-name seed, writes the modeling input and original row indices, and compares the resulting values with the bundled table. The source URL, sizes, expected sample hash and previously verified upstream hash are in `pmlb/manifest.json`. No source file above 100 MB is stored in this repository.
