# casmi-baseline

A small, **dependency-free, CPU-only** spectral-library-search baseline and
local scorer for identifying small molecules from tandem mass spectra (MS/MS).
It was written for the
[Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
Kaggle competition. Teams submit up to 25 ranked SMILES per molecule and are
scored by MRR@25 on the tautomer-canonical InChIKey connectivity block. The
entry deadline is 2026-12-07.

It is **one Python file** ([`casmi_baseline.py`](casmi_baseline.py)), standard
library only, so you can paste it into an offline Kaggle notebook cell or import
it locally.

| Piece | What it does |
| --- | --- |
| `neutral_mass`, `target_mass` | Precursor m/z + adduct → neutral mass (16 common adducts, bracketed or bare names); median over a molecule's spectra. |
| `clean_peaks` | Drop the precursor region and noise (<1% of base peak), centroid within 0.02 Da, normalise. |
| `entropy_similarity` | Weighted **spectral entropy similarity** (Li et al., *Nature Methods* 2021), closed form, one-to-one peak matching. |
| `SpectralLibrary`, `rank_candidates` | Mass-indexed library (bisect, ppm window). Each candidate's score is the mean over the query's spectra of its best same-polarity match. Ties break by mass error. |
| `reciprocal_rank`, `mrr_at_k` | The competition metric for local validation (blank and repeated keys don't use a rank). |
| `rdkit_key14` | Optional (needs RDKit): tautomer-canonical InChIKey first block, the match key. |
| `format_cell`, `write_submission`, `validate_submission` | `id,SMILES;SMILES;…` rows in sample order, deduplicated, never empty, ≤25 candidates. |
| `resolve_columns` | Map standard names to the real parquet column names (case-insensitive aliases). |

## What this is, honestly

This is a **transparent floor**, not a leaderboard contender. Pure library
search can only rank structures that already have reference spectra. The
competition's test molecules are mostly absent from public libraries, so strong
entries add a large structure pool (for example COCONUT), learned
spectrum-to-fingerprint models, fragmentation evidence and a learned ranker. The
public Apache-2.0 Kaggle pipeline by prvsiyan and haideptry, and repos built on
it such as [shiferaxa/casmi26](https://github.com/shiferaxa/casmi26), do all of
that. Use this file to:

- get a valid, explainable baseline submission running in minutes;
- score your own validation split with a dependency-free MRR@25;
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

On Kaggle, [`examples/kaggle_notebook.py`](examples/kaggle_notebook.py) prints
every input file and its columns, builds a library from training spectra near
the test target masses, ranks and writes `submission.csv`, and validates it
against the sample submission. **It has not yet been run on the competition
data.** The parquet schema is only documented on Kaggle, so check the printed
column mapping and adjust `ALIASES` if a name resolves to `None`.

## Tests

```bash
python test_casmi_baseline.py   # or: pytest
```

14 tests on synthetic spectra cover the adduct masses, cleaning, entropy
similarity (identity, disjoint, partial, tolerance, symmetry), the ppm index,
ranking and polarity, the metric, and the submission round trip.

## References

- Li, Y. et al. *Spectral entropy outperforms MS/MS dot product similarity for
  small-molecule compound identification.* Nature Methods 18, 1524–1531 (2021).
- Competition metric as described on the Kaggle competition page: MRR@25 on the
  first 14 characters of the InChIKey after RDKit tautomer canonicalisation.

## Who made this

Made by **MV2 of Munim, Inc.**, with the code written by AI agents working in his
name. If you use it in a competition, disclose AI-assisted work wherever that
competition's rules require it.

## License

MIT — see [LICENSE](LICENSE).
