import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from casmi_baseline import (
    FORMATE,
    H2O,
    PROTON,
    TEST_ADDUCTS,
    SpectralLibrary,
    adduct_polarity,
    clean_peaks,
    entropy_similarity,
    evaluate,
    format_cell,
    mrr_at_k,
    near_any,
    neutral_mass,
    normalize_adduct,
    rank_candidates,
    read_sample_submission,
    reciprocal_rank,
    resolve_columns,
    target_mass,
    validate_submission,
    weight_peaks,
    write_submission,
)

GLUCOSE = 180.063389  # C6H12O6 monoisotopic mass


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_neutral_mass_for_common_adducts():
    assert approx(neutral_mass(181.070665, "[M+H]+"), GLUCOSE)
    assert approx(neutral_mass(179.056113, "[M-H]-"), GLUCOSE)
    assert approx(neutral_mass(GLUCOSE + 22.989218, "[M+Na]+"), GLUCOSE)
    assert approx(neutral_mass((GLUCOSE + 2 * 1.007276) / 2, "[M+2H]2+"), GLUCOSE)
    assert approx(neutral_mass(2 * GLUCOSE + 1.007276, "[2M+H]+"), GLUCOSE)
    assert neutral_mass(181.07, "weird") is None


def test_every_competition_test_adduct_is_known():
    assert len(TEST_ADDUCTS) == 10
    for adduct in TEST_ADDUCTS:
        assert normalize_adduct(adduct) == adduct
        assert neutral_mass(GLUCOSE, adduct) is not None
    assert [adduct_polarity(a) for a in TEST_ADDUCTS] == [1] * 6 + [-1] * 4


def test_neutral_mass_for_double_water_loss_and_formate():
    assert approx(neutral_mass(GLUCOSE + PROTON - 2 * H2O, "[M-2H2O+H]+"), GLUCOSE)
    assert approx(neutral_mass(GLUCOSE + PROTON - 2 * H2O, "[M+H-2H2O]+"), GLUCOSE)
    assert approx(neutral_mass(GLUCOSE + FORMATE, "[M+CH2O2-H]-"), GLUCOSE)
    assert approx(neutral_mass(GLUCOSE + FORMATE, "[M+FA-H]-"), GLUCOSE)
    assert normalize_adduct("M+CH2O2-H") == "[M+CH2O2-H]-"


def test_normalize_adduct_forms():
    assert normalize_adduct("[M+H]+") == "[M+H]+"
    assert normalize_adduct("M+H") == "[M+H]+"
    assert normalize_adduct("M+H+") == "[M+H]+"
    assert normalize_adduct("[M+H]") == "[M+H]+"
    assert normalize_adduct(" [M - H] - ") == "[M-H]-"
    assert normalize_adduct("M-H") == "[M-H]-"
    assert normalize_adduct("M+2H") == "[M+2H]2+"
    assert normalize_adduct("") is None
    assert normalize_adduct(None) is None


def test_target_mass_is_median_and_skips_unknown_adducts():
    spectra = [
        {"precursor_mz": 181.070665, "adduct": "[M+H]+"},
        {"precursor_mz": GLUCOSE + 22.989218, "adduct": "[M+Na]+"},
        {"precursor_mz": 999.0, "adduct": "??"},
    ]
    assert approx(target_mass(spectra), GLUCOSE)
    assert target_mass([{"precursor_mz": 1.0, "adduct": "??"}]) is None


def test_near_any():
    targets = [100.0, GLUCOSE, 300.0]
    assert near_any(GLUCOSE + 0.001, targets, 10.0)
    assert not near_any(GLUCOSE + 0.01, targets, 10.0)
    assert not near_any(400.0, targets, 10.0)
    assert not near_any(None, targets, 10.0)
    assert not near_any(GLUCOSE, [], 10.0)


def test_clean_peaks_merges_filters_and_normalises():
    mzs = [50.0, 50.005, 100.0, 150.0, 179.5, 200.0]
    ints = [10, 10, 100, 0.5, 50, 30]
    peaks = clean_peaks(mzs, ints, precursor_mz=181.0)
    # 179.5 and 200 are in the precursor region; 150 is below 1% of base
    assert len(peaks) == 2
    assert approx(peaks[0][0], 50.0025)
    assert approx(peaks[0][1], 20 / 120)
    assert approx(peaks[1][0], 100.0)
    assert approx(sum(p for _, p in peaks), 1.0)
    assert clean_peaks([], []) == []
    assert clean_peaks([10.0], [0]) == []


def test_weight_peaks_flattens_low_entropy_spectra():
    out = weight_peaks([(100.0, 0.9), (200.0, 0.1)])
    assert out[1][1] > 0.1
    assert approx(sum(p for _, p in out), 1.0)


def test_entropy_similarity_identity_disjoint_and_partial():
    a = [(100.0, 0.5), (200.0, 0.5)]
    b = [(100.0, 0.5), (300.0, 0.5)]
    c = [(400.0, 1.0)]
    lopsided = [(100.0, 0.8), (150.0, 0.15), (200.0, 0.05)]
    assert approx(entropy_similarity(a, a), 1.0)
    assert approx(entropy_similarity(lopsided, lopsided), 1.0)
    assert entropy_similarity(a, c) == 0.0
    assert approx(entropy_similarity(a, b, weighted=False), 0.5)
    assert approx(entropy_similarity(a, b), 0.5)
    assert approx(entropy_similarity(a, b), entropy_similarity(b, a))
    assert entropy_similarity(a, []) == 0.0


def test_entropy_similarity_tolerance():
    a = [(100.0, 1.0)]
    assert approx(entropy_similarity(a, [(100.015, 1.0)]), 1.0)
    assert entropy_similarity(a, [(100.05, 1.0)]) == 0.0


def test_library_within_ppm():
    lib = SpectralLibrary()
    for key, mass in [("a", 100.0), ("b", 180.0634), ("c", 180.0640), ("d", 250.0)]:
        lib.add(key, "C", mass)
    hits = lib.within(GLUCOSE, 10.0)
    assert sorted(r[1] for r in hits) == ["b", "c"]
    assert len(lib) == 4


P1 = [(85.03, 0.6), (127.04, 0.4)]
P2 = [(60.02, 0.5), (91.05, 0.5)]


def _library():
    lib = SpectralLibrary()
    lib.add("GLC", "OCC1OC(O)C(O)C(O)C1O", GLUCOSE, P1, polarity=1)
    lib.add("OTHER", "CCCC", GLUCOSE, P2, polarity=1)
    lib.add("NOSPEC", "CCCO", GLUCOSE + 0.0002)
    lib.add("FAR", "CCCCO", 200.0, P1, polarity=1)
    return lib


def test_rank_candidates_orders_by_similarity_then_mass_error():
    query = [{"precursor_mz": 181.070665, "adduct": "[M+H]+", "peaks": P1}]
    ranked = rank_candidates(query, _library(), ppm=10.0)
    assert [r[0] for r in ranked] == ["GLC", "OTHER", "NOSPEC"]
    assert approx(ranked[0][2], 1.0)
    assert ranked[1][2] == 0.0
    assert [r[0] for r in rank_candidates(query, _library(), k=1)] == ["GLC"]


def test_rank_candidates_respects_polarity_and_unknown_adduct():
    lib = SpectralLibrary()
    lib.add("NEG", "C", GLUCOSE, P1, polarity=-1)
    query = [{"precursor_mz": 181.070665, "adduct": "[M+H]+", "peaks": P1}]
    assert rank_candidates(query, lib)[0][2] == 0.0
    assert rank_candidates([{"precursor_mz": 181.07, "adduct": "??", "peaks": P1}], lib) == []


def test_rank_candidates_with_formate_adduct():
    lib = SpectralLibrary()
    lib.add("GLC", "OCC1OC(O)C(O)C(O)C1O", GLUCOSE, P1, polarity=-1)
    query = [{"precursor_mz": GLUCOSE + FORMATE, "adduct": "[M+CH2O2-H]-", "peaks": P1}]
    assert [r[0] for r in rank_candidates(query, lib)] == ["GLC"]


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank("A", ["A"]) == 1.0
    assert reciprocal_rank("A", ["B", "A"]) == 0.5
    assert reciprocal_rank("A", ["B", "B", "", None, "A"]) == 0.5
    assert reciprocal_rank("A", ["B", "C"]) == 0.0
    assert reciprocal_rank("A", ["B", "A"], k=1) == 0.0
    assert mrr_at_k({"m1": "A", "m2": "B"}, {"m1": ["A"]}) == 0.5
    assert mrr_at_k({}, {}) == 0.0


def test_evaluate_scores_a_labelled_split():
    q1 = [{"precursor_mz": 181.070665, "adduct": "[M+H]+", "peaks": P1}]
    q2 = [{"precursor_mz": 181.070665, "adduct": "[M+H]+", "peaks": P2}]
    lib = _library()
    assert approx(evaluate({"a": q1}, {"a": "GLC"}, lib), 1.0)
    assert approx(evaluate({"a": q1}, {"a": "OTHER"}, lib), 0.5)
    # missing query for a truth id scores 0
    assert approx(evaluate({"a": q1, "b": q2}, {"a": "GLC", "b": "OTHER", "c": "X"}, lib), 2 / 3)


def test_format_cell_dedups_caps_and_never_empty():
    assert format_cell(["CCO", "cco", "CCN"], key=lambda s: s.upper()) == "CCO;CCN"
    assert format_cell(["CCO", "", " ", "a;b", "CCN"]) == "CCO;CCN"
    assert format_cell(["C%d" % i for i in range(30)], k=25).count(";") == 24
    assert format_cell([]) == "C"
    assert format_cell(["CCO"], key=lambda s: None) == "C"


def test_write_read_validate_roundtrip():
    d = tempfile.mkdtemp()
    sample = os.path.join(d, "sample_submission.csv")
    with open(sample, "w", encoding="utf-8", newline="") as fh:
        fh.write("molecule_id,smiles\nm1,C\nm2,C\n")
    header, ids = read_sample_submission(sample)
    assert header == ["molecule_id", "smiles"]
    assert ids == ["m1", "m2"]
    out = os.path.join(d, "submission.csv")
    assert write_submission(out, header, ids, {"m1": ["CCO", "CCN"]}) == 2
    with open(out, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines == ["molecule_id,smiles", "m1,CCO;CCN", "m2,C"]
    assert validate_submission(out, sample) == []
    bad = os.path.join(d, "bad.csv")
    with open(bad, "w", encoding="utf-8", newline="") as fh:
        fh.write("molecule_id,smiles\nm2,C\nm1,C\n")
    assert validate_submission(bad, sample) != []


def test_resolve_columns_case_insensitive():
    cols = ["Precursor_MZ", "ms2_mzs", "other"]
    aliases = {
        "precursor_mz": ["precursor_mz"],
        "mzs": ["mzs", "ms2_mzs"],
        "adduct": ["adduct", "precursor_type"],
    }
    assert resolve_columns(cols, aliases) == {
        "precursor_mz": "Precursor_MZ",
        "mzs": "ms2_mzs",
        "adduct": None,
    }


def test_resolve_columns_on_the_competition_schema():
    train = ["ingest_lib", "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
             "ionization_mode", "instrument_type", "adduct", "adduct_orig", "precursor_mz",
             "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities", "num_peaks",
             "base_peak_intensity", "collision_energy_ev", "collision_energy_orig",
             "collision_energy_orig_units"]
    aliases = {
        "key": ["inchikey14", "inchikey"],
        "intensities": ["ms2_normalized_intensities", "ms2_intensities", "intensities"],
        "molecule_id": ["molecule_id"],
    }
    assert resolve_columns(train, aliases) == {
        "key": "inchikey14",
        "intensities": "ms2_normalized_intensities",
        "molecule_id": None,
    }


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_casmi_baseline OK")
