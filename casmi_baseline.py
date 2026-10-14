"""casmi_baseline -- a small, dependency-free spectral-library-search baseline
and local scorer for small-molecule identification from tandem mass spectra
(MS/MS), written for the Enveda CASMI 2026 Kaggle competition.

It is one file on purpose: paste it into a Kaggle notebook cell (code
competitions run offline) or import it locally.  Pieces:

* ``neutral_mass`` / ``target_mass`` -- precursor m/z + adduct -> neutral mass;
  the median over a molecule's spectra is its target mass.  The table covers
  all ten adducts used in the competition's test set.
* ``clean_peaks`` -- drop precursor-region and noise peaks, centroid, normalise.
* ``entropy_similarity`` -- (weighted) spectral entropy similarity
  (Li et al., Nature Methods 2021).
* ``SpectralLibrary`` / ``rank_candidates`` -- precursor-mass-indexed library
  search that ranks candidate structures for one molecule.
* ``reciprocal_rank`` / ``mrr_at_k`` / ``evaluate`` -- the competition metric
  (MRR@25 over structure keys) for local validation.
* ``format_cell`` / ``write_submission`` / ``validate_submission`` -- the
  ``molecule_id,<up to 25 SMILES joined by ';'>`` submission file.
* ``rdkit_key14`` -- optional, needs RDKit: tautomer-canonical InChIKey
  connectivity block (first 14 characters), the competition's match key.

Made by MV2 of Munim, Inc., with the code written by AI agents working in his
name.  MIT licensed.
"""

import bisect
import csv
import math
import statistics

__version__ = "0.2.0"

PROTON = 1.007276
H2O = 18.010565
FORMATE = 44.998201  # HCOO- (CH2O2 - H)

# adduct -> (molecules per ion, charge, mass added to n*M); m/z = (n*M + shift) / |z|
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+Na]+": (1, 1, 22.989218),
    "[M+NH4]+": (1, 1, 18.033823),
    "[M+K]+": (1, 1, 38.963158),
    "[M+H-H2O]+": (1, 1, PROTON - H2O),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M+H-2H2O]+": (1, 1, PROTON - 2 * H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, 22.989218),
    "[M]+": (1, 1, -0.000549),
    "[M-H]-": (1, -1, -PROTON),
    "[M-H-H2O]-": (1, -1, -PROTON - H2O),
    "[M-H2O-H]-": (1, -1, -PROTON - H2O),
    "[M+Cl]-": (1, -1, 34.969402),
    "[M+FA-H]-": (1, -1, FORMATE),
    "[M+HCOO]-": (1, -1, FORMATE),
    "[M+CH2O2-H]-": (1, -1, FORMATE),
    "[M+CH3COO]-": (1, -1, 59.013851),
    "[M-2H]2-": (1, -2, -2 * PROTON),
    "[2M-H]-": (2, -1, -PROTON),
}

# the ten adducts listed for the CASMI 2026 test spectra
TEST_ADDUCTS = (
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
)


# ---------------------------------------------------------------- masses


def normalize_adduct(adduct):
    """Return the canonical ``ADDUCTS`` key for ``adduct``, or None.

    Accepts the bracketed form (``"[M+H]+"``), brackets without a charge
    (``"[M+H]"``) and the bare form (``"M+H"``, ``"M+H+"``), ignoring spaces."""
    if not adduct:
        return None
    key = str(adduct).replace(" ", "")
    if key in ADDUCTS:
        return key
    if key.startswith("["):
        core = key[1:key.index("]")] if "]" in key else key[1:]
    else:
        core = key.rstrip("+-")
    if core:
        for suffix in ("+", "-", "2+", "2-"):
            candidate = "[%s]%s" % (core, suffix)
            if candidate in ADDUCTS:
                return candidate
    return None


def adduct_polarity(adduct):
    """+1 / -1 for a known adduct, else 0."""
    key = normalize_adduct(adduct)
    if key is None:
        return 0
    return 1 if ADDUCTS[key][1] > 0 else -1


def neutral_mass(precursor_mz, adduct):
    """Neutral monoisotopic mass of the molecule behind a precursor ion, or
    None if the adduct is unknown."""
    key = normalize_adduct(adduct)
    if key is None or precursor_mz is None:
        return None
    n, z, shift = ADDUCTS[key]
    return (float(precursor_mz) * abs(z) - shift) / n


def target_mass(spectra):
    """Median neutral mass over a molecule's spectra.

    ``spectra`` is an iterable of dicts with ``precursor_mz`` and ``adduct``.
    Spectra with an unknown adduct are skipped; returns None if none remain."""
    masses = []
    for s in spectra:
        m = neutral_mass(s.get("precursor_mz"), s.get("adduct"))
        if m is not None:
            masses.append(m)
    return statistics.median(masses) if masses else None


def ppm_window(mass, ppm):
    """``(low, high)`` mass bounds for a +/- ``ppm`` tolerance."""
    delta = mass * ppm * 1e-6
    return mass - delta, mass + delta


def near_any(mass, sorted_targets, ppm):
    """True if ``mass`` is within ``ppm`` of any value in ``sorted_targets``."""
    if mass is None:
        return False
    lo, hi = ppm_window(mass, ppm)
    i = bisect.bisect_left(sorted_targets, lo)
    return i < len(sorted_targets) and sorted_targets[i] <= hi


# ---------------------------------------------------------------- spectra


def clean_peaks(mzs, intensities, precursor_mz=None, noise=0.01,
                precursor_margin=1.6, centroid_tol=0.02):
    """Return a cleaned spectrum as a sorted list of ``(mz, intensity)`` with
    intensities summing to 1.

    Drops non-positive peaks and peaks within ``precursor_margin`` Da below the
    precursor (or above it), merges peaks closer than ``centroid_tol`` Da
    (intensity-weighted m/z, summed intensity), then removes peaks under
    ``noise`` x the base peak."""
    peaks = []
    for m, i in zip(mzs, intensities):
        m, i = float(m), float(i)
        if i <= 0:
            continue
        if precursor_mz is not None and m > float(precursor_mz) - precursor_margin:
            continue
        peaks.append((m, i))
    if not peaks:
        return []
    peaks.sort()
    merged = [peaks[0]]
    for m, i in peaks[1:]:
        pm, pi = merged[-1]
        if m - pm < centroid_tol:
            total = pi + i
            merged[-1] = ((pm * pi + m * i) / total, total)
        else:
            merged.append((m, i))
    base = max(i for _, i in merged)
    kept = [(m, i) for m, i in merged if i >= noise * base]
    total = sum(i for _, i in kept)
    return [(m, i / total) for m, i in kept]


def spectral_entropy(peaks):
    """Shannon entropy (nats) of a normalised spectrum."""
    return -sum(p * math.log(p) for _, p in peaks if p > 0)


def weight_peaks(peaks):
    """Entropy-based intensity weighting from Li et al. 2021: spectra with
    entropy below 3 get intensities raised to ``0.25 + 0.25 * S`` and are
    re-normalised."""
    s = spectral_entropy(peaks)
    if s >= 3 or not peaks:
        return list(peaks)
    w = 0.25 + 0.25 * s
    raised = [(m, p ** w) for m, p in peaks]
    total = sum(p for _, p in raised)
    return [(m, p / total) for m, p in raised]


def _xlogx(x):
    return x * math.log(x) if x > 0 else 0.0


def entropy_similarity(a, b, tol=0.02, weighted=True):
    """Spectral entropy similarity in ``[0, 1]`` between two cleaned spectra
    (outputs of :func:`clean_peaks`).

    Uses the closed form ``sum over matched peak pairs of
    [f(x+y) - f(x) - f(y)] / ln 4`` with ``f(x) = x ln x``, which equals
    ``1 - (2 S_AB - S_A - S_B) / ln 4``.  Peaks are matched one-to-one within
    ``tol`` Da, closest pairs first."""
    if weighted:
        a, b = weight_peaks(a), weight_peaks(b)
    if not a or not b:
        return 0.0
    pairs = []
    start = 0
    for ia, (ma, _) in enumerate(a):
        while start < len(b) and b[start][0] < ma - tol:
            start += 1
        j = start
        while j < len(b) and b[j][0] <= ma + tol:
            pairs.append((abs(b[j][0] - ma), ia, j))
            j += 1
    pairs.sort()
    used_a, used_b = set(), set()
    total = 0.0
    for _, ia, jb in pairs:
        if ia in used_a or jb in used_b:
            continue
        used_a.add(ia)
        used_b.add(jb)
        x, y = a[ia][1], b[jb][1]
        total += _xlogx(x + y) - _xlogx(x) - _xlogx(y)
    return min(1.0, max(0.0, total / math.log(4)))


# ---------------------------------------------------------------- search


class SpectralLibrary:
    """Reference structures indexed by neutral mass.

    Add structures with :meth:`add`; ``peaks`` may be None for a structure
    known only by its mass (a candidate with no reference spectrum)."""

    def __init__(self):
        self._rows = []  # (mass, key, smiles, polarity, peaks)
        self._masses = []
        self._dirty = False

    def __len__(self):
        return len(self._rows)

    def add(self, key, smiles, mass, peaks=None, polarity=0):
        self._rows.append((float(mass), key, smiles, polarity, peaks))
        self._dirty = True

    def _index(self):
        if self._dirty:
            self._rows.sort(key=lambda r: r[0])
            self._masses = [r[0] for r in self._rows]
            self._dirty = False

    def within(self, mass, ppm):
        """Rows ``(mass, key, smiles, polarity, peaks)`` within ``ppm`` of ``mass``."""
        self._index()
        lo, hi = ppm_window(mass, ppm)
        i = bisect.bisect_left(self._masses, lo)
        j = bisect.bisect_right(self._masses, hi)
        return self._rows[i:j]


def rank_candidates(query_spectra, library, ppm=10.0, k=25, tol=0.02,
                    weighted=True):
    """Rank candidate structures for one molecule.

    ``query_spectra`` is a list of dicts with ``precursor_mz``, ``adduct`` and
    ``peaks`` (cleaned).  Candidates are library structures within ``ppm`` of
    the molecule's target mass.  A candidate's score is the mean, over query
    spectra, of the best entropy similarity to any of its reference spectra of
    the same polarity (0 if it has none).  Ties break by mass error, then key.

    Returns up to ``k`` tuples ``(key, smiles, score)`` with unique keys."""
    mass = target_mass(query_spectra)
    if mass is None:
        return []
    by_key = {}
    for row_mass, key, smiles, polarity, peaks in library.within(mass, ppm):
        entry = by_key.setdefault(key, {"smiles": smiles, "err": abs(row_mass - mass), "refs": []})
        entry["err"] = min(entry["err"], abs(row_mass - mass))
        if peaks:
            entry["refs"].append((polarity, peaks))
    queries = [(adduct_polarity(q.get("adduct")), q.get("peaks") or []) for q in query_spectra]
    ranked = []
    for key, entry in by_key.items():
        score = 0.0
        if entry["refs"] and queries:
            best_sum = 0.0
            for q_pol, q_peaks in queries:
                best = 0.0
                for r_pol, r_peaks in entry["refs"]:
                    if q_pol and r_pol and q_pol != r_pol:
                        continue
                    best = max(best, entropy_similarity(q_peaks, r_peaks, tol, weighted))
                best_sum += best
            score = best_sum / len(queries)
        ranked.append((-score, entry["err"], key, entry["smiles"]))
    ranked.sort()
    return [(key, smiles, -neg) for neg, _, key, smiles in ranked[:k]]


# ---------------------------------------------------------------- metric


def reciprocal_rank(truth_key, candidate_keys, k=25):
    """1/rank of ``truth_key`` among ``candidate_keys`` (0 if not in the top
    ``k``).  Empty and repeated keys do not use up a rank."""
    seen = set()
    rank = 0
    for c in candidate_keys:
        if not c or c in seen:
            continue
        seen.add(c)
        rank += 1
        if rank > k:
            break
        if c == truth_key:
            return 1.0 / rank
    return 0.0


def mrr_at_k(truth, predictions, k=25):
    """Mean reciprocal rank over every id in ``truth`` (``id -> key``);
    ``predictions`` maps ``id -> ranked keys``.  Missing ids score 0."""
    if not truth:
        return 0.0
    return sum(reciprocal_rank(t, predictions.get(i, ()), k) for i, t in truth.items()) / len(truth)


def evaluate(queries, truth, library, ppm=10.0, k=25, tol=0.02, weighted=True):
    """MRR@k of :func:`rank_candidates` on a labelled split.

    ``queries`` maps ``id -> query spectra``, ``truth`` maps ``id -> key``.
    Keep the query spectra out of ``library`` (or the score is meaningless)."""
    predictions = {
        i: [key for key, _, _ in rank_candidates(q, library, ppm, k, tol, weighted)]
        for i, q in queries.items()
    }
    return mrr_at_k(truth, predictions, k)


_TAUTOMERS = None


def rdkit_key14(smiles):
    """First 14 characters of the InChIKey after RDKit tautomer
    canonicalisation (the competition's match key), or None if unparsable.
    Requires RDKit."""
    global _TAUTOMERS
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if _TAUTOMERS is None:
        _TAUTOMERS = rdMolStandardize.TautomerEnumerator()
    try:
        mol = _TAUTOMERS.Canonicalize(mol)
    except Exception:
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


# ---------------------------------------------------------------- submission


def format_cell(smiles_list, key=None, k=25, placeholder="C"):
    """Join up to ``k`` SMILES with ``';'``, skipping blanks and structures
    whose ``key(smiles)`` is None or already used.  Never returns an empty
    cell (falls back to ``placeholder``)."""
    out = []
    seen = set()
    for s in smiles_list:
        s = (s or "").strip()
        if not s or ";" in s:
            continue
        kk = key(s) if key else s
        if kk is None or kk in seen:
            continue
        seen.add(kk)
        out.append(s)
        if len(out) == k:
            break
    return ";".join(out) if out else placeholder


def read_sample_submission(path):
    """Return ``(header, ids)`` from the competition's sample submission."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ids = [row[0] for row in reader if row]
    return header, ids


def write_submission(path, header, ids, predictions, key=None, k=25):
    """Write one row per id, in the given order: ``id, cell``.

    ``header`` is the sample submission's header (first two columns are used);
    ``predictions`` maps ``id -> ranked SMILES``.  Returns rows written."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header[:2])
        for i in ids:
            writer.writerow([i, format_cell(predictions.get(i, ()), key, k)])
    return len(ids)


def validate_submission(path, sample_path, k=25):
    """Return a list of problems (empty if the file looks valid)."""
    problems = []
    s_header, s_ids = read_sample_submission(sample_path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ["file is empty"]
    if rows[0][:2] != s_header[:2]:
        problems.append("header %r != sample %r" % (rows[0][:2], s_header[:2]))
    body = [r for r in rows[1:] if r]
    if [r[0] for r in body] != s_ids:
        problems.append("ids differ from the sample (count or order)")
    for r in body:
        cands = [c for c in (r[1] if len(r) > 1 else "").split(";") if c.strip()]
        if not 1 <= len(cands) <= k:
            problems.append("id %s has %d candidates" % (r[0], len(cands)))
    return problems


# ---------------------------------------------------------------- columns


def resolve_columns(columns, aliases):
    """Map standard names to the first matching actual column name (or None).

    ``aliases`` is ``{standard_name: [candidate column names, ...]}``; matching
    is case-insensitive."""
    lower = {str(c).lower(): c for c in columns}
    return {
        name: next((lower[c.lower()] for c in cands if c.lower() in lower), None)
        for name, cands in aliases.items()
    }
