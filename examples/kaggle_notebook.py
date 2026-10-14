"""Run the casmi_baseline library search inside a Kaggle notebook (offline).

Usage in a notebook: paste casmi_baseline.py into the first cell (or attach it
as a dataset), then run this file's contents in the next cell.

The competition's parquet column names are only documented on Kaggle, so the
first thing this script does is print every input file and its columns.  If a
standard name below resolves to None, add the real column name to ALIASES and
re-run.  This script has not yet been run against the competition data.
"""

import bisect
import glob
import os

import pandas as pd

from casmi_baseline import (
    SpectralLibrary,
    adduct_polarity,
    clean_peaks,
    neutral_mass,
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

ALIASES = {
    "molecule_id": ["molecule_id", "query_id", "id", "compound_id", "target_id"],
    "spectrum_id": ["spectrum_id", "spec_id", "scan_id"],
    "smiles": ["normalized_smiles", "smiles", "canonical_smiles"],
    "inchikey": ["inchikey", "inchi_key"],
    "precursor_mz": ["precursor_mz", "precursormz", "pepmass", "prec_mz"],
    "adduct": ["adduct", "precursor_type", "ion_type"],
    "mzs": ["ms2_mzs", "mzs", "mz", "peaks_mz"],
    "intensities": ["ms2_intensities", "intensities", "intensity", "peaks_intensity"],
}


def show_inputs():
    files = sorted(glob.glob(os.path.join(INPUT, "**", "*.*"), recursive=True))
    for path in files:
        print(path)
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
            print("   rows=%d columns=%s" % (len(df), list(df.columns)))
    return files


def find(files, *words, ext=".parquet"):
    for path in files:
        name = os.path.basename(path).lower()
        if path.endswith(ext) and all(w in name for w in words):
            return path
    return None


def spectrum(row, col):
    peaks = clean_peaks(row[col["mzs"]], row[col["intensities"]], row[col["precursor_mz"]])
    return {"precursor_mz": row[col["precursor_mz"]], "adduct": row[col["adduct"]], "peaks": peaks}


def main():
    files = show_inputs()
    sample_path = find(files, "sample", ext=".csv")
    train_path, test_path = find(files, "train"), find(files, "test")
    print("sample:", sample_path, "\ntrain:", train_path, "\ntest:", test_path)

    test = pd.read_parquet(test_path)
    tcol = resolve_columns(test.columns, ALIASES)
    print("test columns:", tcol)
    queries = {}
    for _, row in test.iterrows():
        queries.setdefault(str(row[tcol["molecule_id"]]), []).append(spectrum(row, tcol))
    targets = sorted(m for m in (target_mass(q) for q in queries.values()) if m is not None)
    print("molecules:", len(queries), "with a target mass:", len(targets))

    # keep only training spectra whose neutral mass is near some test target
    train = pd.read_parquet(train_path)
    col = resolve_columns(train.columns, ALIASES)
    print("train columns:", col)
    lib = SpectralLibrary()
    for _, row in train.iterrows():
        mass = neutral_mass(row[col["precursor_mz"]], row[col["adduct"]])
        if mass is None:
            continue
        i = bisect.bisect_left(targets, mass * (1 - PPM * 1e-6))
        if i == len(targets) or targets[i] > mass * (1 + PPM * 1e-6):
            continue
        smiles = row[col["smiles"]]
        key = str(row[col["inchikey"]])[:14] if col["inchikey"] else smiles
        lib.add(key, smiles, mass, spectrum(row, col)["peaks"], adduct_polarity(row[col["adduct"]]))
    print("library spectra kept:", len(lib))

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
