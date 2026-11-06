"""Turn a downloaded benchmark file into a dataset folder `hbf_eval` can read (see hbf_eval/data.py).

    python3 tools/prepare_dataset.py sift-128-euclidean.hdf5 data/sift --dtype uint8 --metric l2
    python3 tools/prepare_dataset.py --parquet-base train-00001-of-00009.parquet --parquet-query test-00001-of-00001.parquet \\
        data/deep --dtype float32 --metric angular --max-base 200000

Writes <out>/base.npy, <out>/query.npy, <out>/meta.json. The folder name is the dataset name the tasks ask for: `sift` (128-d uint8, L2)
and `deep` (96-d float32). Then: python3 -m hbf_eval.bench run --data-root data ...

Input is an ann-benchmarks style HDF5 file (datasets `train` and `test`) or parquet files with an `emb` list column.
Needs h5py (HDF5) or pyarrow (parquet), which the evaluator itself does not.
"""
import argparse
import json
import os

import numpy as np


def parquet_rows(path, limit=None):
    import pyarrow.parquet as pq
    col = pq.ParquetFile(path).read(columns=["emb"]).column("emb").combine_chunks()
    return np.asarray(col.values.to_numpy(), dtype=np.float32).reshape(len(col), -1)[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--parquet-base")
    ap.add_argument("--parquet-query")
    ap.add_argument("--dtype", choices=("uint8", "float32"), required=True)
    ap.add_argument("--metric", choices=("l2", "angular"), required=True)
    ap.add_argument("--max-base", type=int, default=None, help="keep only the first N base vectors")
    a = ap.parse_args()
    if a.hdf5:
        import h5py
        f = h5py.File(a.hdf5)
        base, query = np.array(f["train"][:a.max_base]), np.array(f["test"])
    elif a.parquet_base and a.parquet_query:
        base, query = parquet_rows(a.parquet_base, a.max_base), parquet_rows(a.parquet_query)
    else:
        ap.error("give an HDF5 file, or --parquet-base and --parquet-query")
    if a.dtype == "uint8":
        if not (np.all(base == np.round(base)) and base.min() >= 0 and base.max() <= 255):
            ap.error("base vectors are not integers in 0..255; use --dtype float32")
        base = base.astype(np.uint8)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "base.npy"), base)
    np.save(os.path.join(a.out, "query.npy"), query.astype(np.float32))
    json.dump({"metric": a.metric, "dtype": a.dtype}, open(os.path.join(a.out, "meta.json"), "w"))
    print("wrote %s: base %s %s, query %s" % (a.out, base.shape, base.dtype, query.shape))


if __name__ == "__main__":
    main()
