# casmi-baseline

A small, **dependency-free, CPU-only** spectral-library-search baseline and
local scorer for identifying small molecules from tandem mass spectra (MS/MS).
It was written for the
[Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
Kaggle competition. Teams submit up to 25 ranked SMILES per molecule and are
scored by MRR@25 on the tautomer-canonical InChIKey connectivity block. The
entry deadline is 2026-12-07; final submissions are due 2026-12-14.

It is **one Python file** ([`casmi_baseline.py`](casmi_baseline.py)), standard
library only, so you can paste it into an offline Kaggle notebook cell or import
it locally.

| Piece | What it does |
| --- | --- |
| `neutral_mass`, `target_mass` | Precursor m/z + adduct → neutral mass. Covers all ten test-set adducts (`TEST_ADDUCTS`, including `[M-2H2O+H]+` and `[M+CH2O2-H]-`) plus common training ones, bracketed or bare names; median over a molecule's spectra. |
| `clean_peaks` | Drop the precursor region and noise (<1% of base peak), centroid within 0.02 Da, normalise. |
| `entropy_similarity` | Weighted **spectral entropy similarity** (Li et al., *Nature Methods* 2021), closed form, one-to-one peak matching. |
| `SpectralLibrary`, `rank_candidates`, `near_any` | Mass-indexed library (bisect, ppm window). Each candidate's score is the mean over the query's spectra of its best same-polarity match. Ties break by mass error. |
| `fuse_rankings`, `read_submission` | Weighted **reciprocal-rank fusion** of several ranked candidate lists for a molecule, e.g. a bigger pipeline's output plus this library search. |
| `reciprocal_rank`, `mrr_at_k`, `evaluate` | The competition metric for local validation (blank and repeated keys don't use a rank); `evaluate` scores a labelled split end to end. |
| `rdkit_key14` | Optional (needs RDKit): tautomer-canonical InChIKey first block, the match key. |
| `format_cell`, `write_submission`, `validate_submission` | `molecule_id,SMILES;SMILES;…` rows in sample order, deduplicated, never empty, ≤25 candidates. |
| `resolve_columns` | Map standard names to the real parquet column names (case-insensitive aliases). |

## What this is, honestly

This is a **transparent floor**, not a leaderboard contender. Pure library
search can only rank structures that already have reference spectra, and most
test molecules have none. Public reports put library-only entries at roughly
0.14–0.23 on the public leaderboard. On 2026-10-14 the best public notebooks
(several tied at 0.441, Apache-2.0) add a large structure pool (COCONUT),
analog and fragmentation evidence, a spectrum-to-fingerprint model and a
learned ranker; the top of the board is near 0.49. The Apache-2.0 public
pipeline by prvsiyan and haideptry, and repos built on it such as
[shiferaxa/casmi26](https://github.com/shiferaxa/casmi26), do all of that. Use
this file to:

- get a valid, explainable baseline submission running;
- score your own validation split with a dependency-free MRR@25;
- fuse a stronger pipeline's ranking with spectral-library evidence;
- reuse the adduct table, entropy similarity and submission validator in a
  bigger pipeline.

## Use

```python
from casmi_baseline import SpectralLibrary, clean_peaks, rank_candidates, mrr_at_k

lib = SpectralLibrary()
lib.add(key="WQZGKKKJIJFFOK", smiles="OCC1OC(O)C(O)C(O)C1O", mass=180.063389,
        peaks=clean_peaks(ref_mzs, ref_ints, 181.0707), polarity=+1)

query = [{"precursor_mz": 181.0707, "adduct": "[M+H]+",
          "peaks": clean_peaks(q_mzs, q_ints, 181.0707)}]
for key, smiles, score in rank_candidates(query, lib, ppm=10, k=25):
    print(key, smiles, round(score, 3))
```

### Fusing with a bigger pipeline

Run the other pipeline first in the same notebook so it writes its own
submission file, then blend it with the library search, one molecule at a time:

```python
from casmi_baseline import fuse_rankings, rank_candidates, rdkit_key14, read_submission

theirs = read_submission("/kaggle/working/pipeline_submission.csv")
fused = {}
for mid, smiles_list in theirs.items():
    a = [(rdkit_key14(s), s) for s in smiles_list]
    b = [(rdkit_key14(s), s) for _, s, _ in rank_candidates(queries[mid], lib)]
    fused[mid] = [s for _, s, _ in fuse_rankings([a, b], weights=[1.0, 0.5])]
```

Use the same key function on both lists so that one structure counts as one
candidate. The weights above are placeholders, not tuned values. Tune them on
the `enveda-np-examples` validation split (see `VALIDATE` below) with
`mrr_at_k` before trusting them, because fusion can lower the score when one
list is much weaker.

### On Kaggle

[`examples/kaggle_notebook.py`](examples/kaggle_notebook.py) uses the
competition's column names (`ms2_mzs`, `ms2_normalized_intensities`,
`precursor_mz`, `adduct`, `inchikey14`, `normalized_smiles`, `ingest_lib`,
`precursor_error_ppm`; test rows grouped by `molecule_id`; sample submission
`molecule_id,smiles`). It reads the ~2.5M training spectra in batches with
pyarrow, keeps only spectra within 10 ppm of a test target mass (and with a
labelled precursor error under 20 ppm), ranks, writes `submission.csv` and
validates it against the sample submission.

Set `VALIDATE = True` to score it locally instead. Queries are the
`enveda-np-examples` structures, which the host measured on the test
instrument. Their own spectra are left out of the library, while other
libraries' spectra of the same structures stay in. Public write-ups report that
this split tracks the leaderboard and random held-out training structures
don't.

**The notebook has not yet been run on the competition data.** Check the
printed column mapping before trusting the output. The visible `test.parquet`
is a placeholder; Kaggle swaps in the hidden test set when a submitted notebook
is re-run.

## Changes

- **0.3.0** – Adds `fuse_rankings` (weighted reciprocal-rank fusion, Cormack et
  al. 2009) and `read_submission`, so the library search can be blended with a
  stronger pipeline's ranked candidates. Updates the leaderboard figures to
  2026-10-14. 24 tests.
- **0.2.0** – Adds the two test-set adducts missing from 0.1.0
  (`[M-2H2O+H]+`, `[M+CH2O2-H]-`); without them those molecules got no
  candidates. The notebook now uses the real column names (0.1.0 looked for
  intensities under the wrong name and would have stopped), reads training data
  in batches and has a `VALIDATE` mode. New: `TEST_ADDUCTS`, `near_any`,
  `evaluate`. 20 tests.
- **0.1.0** – First release.

## Tests

```bash
python test_casmi_baseline.py   # or: pytest
```

24 tests on synthetic spectra cover the adduct masses (all ten test-set
adducts), cleaning, entropy similarity (identity, disjoint, partial, tolerance,
symmetry), the ppm index, ranking and polarity, rank fusion (weights, ties,
repeated keys), the metric and split scorer, the submission round trip, and
column resolution on the competition schema.

## References

- Li, Y. et al. *Spectral entropy outperforms MS/MS dot product similarity for
  small-molecule compound identification.* Nature Methods 18, 1524–1531 (2021).
- Cormack, G. V., Clarke, C. L. A. & Büttcher, S. *Reciprocal rank fusion
  outperforms Condorcet and individual rank learning methods.* SIGIR 2009.
- Competition metric as described on the Kaggle competition page: MRR@25 on the
  first 14 characters of the InChIKey after RDKit tautomer canonicalisation.

## Who made this

Made by **MV2 of Munim, Inc.**, with the code written by AI agents working in his
name. If you use it in a competition, disclose AI-assisted work wherever that
competition's rules require it.

## License

Apache-2.0 — see [LICENSE](LICENSE).
