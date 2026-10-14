"""Run the casmi_baseline library search inside a Kaggle notebook (offline).

Usage in a notebook: paste casmi_baseline.py into the first cell (or attach it
as a dataset), then run this file's contents in the next cell.

Column names follow the competition's published schema:

* train.parquet (~2.5M spectra): ingest_lib, normalized_smiles, inchikey,
  inchikey14, adduct, precursor_mz, precursor_error_ppm, ms2_mzs,
  ms2_normalized_intensities, ... (no molecule_id: group by inchikey14)
* test.parquet: molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs,
  ms2_normalized_intensities, ...  The visible file is a placeholder; Kaggle
  swaps in the hidden test set when the submitted notebook is re-run.
* sample_submission.csv: molecule_id,smiles

The training file is read in batches (pyarrow) and only spectra near a target
mass are kept, so memory stays small.  Set VALIDATE = True to score the
baseline locally on the host-measured ``enveda-np-examples`` structures, with
their own spectra left out of the library (other libraries' spectra of the
same structure stay in).  Public write-ups report this split as the one that
tracks the leaderboard.

This script has not yet been run on the competition data; check the printed
column mapping first.
"""

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi_baseline import (
    ADDUCTS,
    SpectralLibrary,
    adduct_polarity,
    clean_peaks,
    evaluate,
    normalize_adduct,
    rank_candidates,
    read_sample_submission,
    resolve_columns,
    target_mass,
    validate_submission,
    write_submission,
)

INPUT = "/kaggle/input"
PPM = 10.0
K = 25
MAX_ERROR_PPM = 20.0  # drop training spectra whose labelled precursor is further off
BATCH = 200_000
VALIDATE = False
VALID_LIB = "enveda-np-examples"

ALIASES = {
    "molecule_id": ["molecule_id"],
    "smiles": ["normalized_smiles", "smiles"],
    "key": ["inchikey14", "inchikey"],
    "precursor_mz": ["precursor_mz"],
    "adduct": ["adduct"],
    "mzs": ["ms2_mzs"],
    "intensities": ["ms2_normalized_intensities", "ms2_intensities", "intensities"],
    "lib": ["ingest_lib"],
    "error_ppm": ["precursor_error_ppm"],
}


def find(files, word, ext):
    for path in files:
        name = os.path.basename(path).lower()
        if path.endswith(ext) and word in name:
            return path
    return None


def columns_of(path):
    col = resolve_columns(pq.ParquetFile(path).schema_arrow.names, ALIASES)
    print(os.path.basename(path), col)
    return col


def batches(path, col):
    """Yield pandas frames of the needed columns, BATCH rows at a time."""
    names = sorted({c for c in col.values() if c})
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=names):
        yield batch.to_pandas()


def neutral_masses(df, col):
    """Neutral mass per row (NaN where the adduct is unknown)."""
    out = np.full(len(df), np.nan)
    mz = df[col["precursor_mz"]].to_numpy(dtype=float)
    adducts = df[col["adduct"]].astype(str).to_numpy()
    for a in np.unique(adducts):
        key = normalize_adduct(a)
        if key is None:
            continue
        n, z, shift = ADDUCTS[key]
        sel = adducts == a
        out[sel] = (mz[sel] * abs(z) - shift) / n
    return out


def near(masses, targets, ppm):
    """Boolean mask: mass within ppm of some value in sorted ``targets``."""
    idx = np.searchsorted(targets, masses * (1 - ppm * 1e-6))
    hit = np.zeros(len(masses), dtype=bool)
    ok = idx < len(targets)
    hit[ok] = targets[idx[ok]] <= masses[ok] * (1 + ppm * 1e-6)
    return hit


def spectra(df, col):
    """``(precursor_mz, adduct, cleaned peaks)`` per row."""
    for prec, adduct, mzs, ints in zip(df[col["precursor_mz"]], df[col["adduct"]],
                                       df[col["mzs"]], df[col["intensities"]]):
        yield prec, adduct, clean_peaks(mzs, ints, prec)


def group_queries(df, col, ids):
    queries = {}
    for gid, (prec, adduct, peaks) in zip(ids, spectra(df, col)):
        queries.setdefault(str(gid), []).append({"precursor_mz": prec, "adduct": adduct, "peaks": peaks})
    return queries


def build_library(train_path, col, targets, exclude_lib=None):
    lib = SpectralLibrary()
    for df in batches(train_path, col):
        keep = near(neutral_masses(df, col), targets, PPM)
        if col["error_ppm"]:
            err = df[col["error_ppm"]].to_numpy(dtype=float)
            keep &= ~(np.abs(err) > MAX_ERROR_PPM)  # NaN errors are kept
        if exclude_lib and col["lib"]:
            keep &= (df[col["lib"]] != exclude_lib).to_numpy()
        sub = df[keep]
        masses = neutral_masses(sub, col)
        keys = sub[col["key"]].astype(str).str[:14]
        for mass, key, smiles, (prec, adduct, peaks) in zip(masses, keys, sub[col["smiles"]], spectra(sub, col)):
            lib.add(key, smiles, mass, peaks, adduct_polarity(adduct))
    print("library spectra kept:", len(lib))
    return lib


def validation_queries(train_path, col):
    hold = pd.concat([df[df[col["lib"]] == VALID_LIB] for df in batches(train_path, col)])
    return group_queries(hold, col, hold[col["key"]].astype(str).str[:14])


def main():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "*.*"), recursive=True))
    sample_path = find(files, "sample", ".csv")
    train_path, test_path = find(files, "train", ".parquet"), find(files, "test", ".parquet")
    print("sample:", sample_path, "\ntrain:", train_path, "\ntest:", test_path)
    col = columns_of(train_path)

    if VALIDATE:
        queries = validation_queries(train_path, col)
        targets = np.array(sorted(m for m in (target_mass(q) for q in queries.values()) if m is not None))
        print("validation structures:", len(queries))
        lib = build_library(train_path, col, targets, exclude_lib=VALID_LIB)
        truth = {k: k for k in queries}
        print("MRR@%d (own spectra hidden): %.4f" % (K, evaluate(queries, truth, lib, PPM, K)))
        return

    tcol = columns_of(test_path)
    test = pq.read_table(test_path).to_pandas()
    queries = group_queries(test, tcol, test[tcol["molecule_id"]])
    targets = np.array(sorted(m for m in (target_mass(q) for q in queries.values()) if m is not None))
    print("molecules:", len(queries), "with a target mass:", len(targets))
    lib = build_library(train_path, col, targets)

    try:
        from casmi_baseline import rdkit_key14
        rdkit_key14("C")
        dedup = rdkit_key14
    except ImportError:
        dedup = None

    preds = {mid: [smiles for _, smiles, _ in rank_candidates(q, lib, PPM, K)]
             for mid, q in queries.items()}
    header, ids = read_sample_submission(sample_path)
    write_submission("submission.csv", header, ids, preds, key=dedup, k=K)
    print("problems:", validate_submission("submission.csv", sample_path, K))


if __name__ == "__main__":
    main()
