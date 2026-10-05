"""Run the stats pipeline on fake data with planted effects and check it recovers them."""

import numpy as np
import pytest

from python.analysis import stats
from python.analysis.make_fake_metrics import generate_fake_metrics
from python.analysis.stats import (WORST, CLEAN, holm, bootstrap_ci, paired_wilcoxon)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    df, truth = generate_fake_metrics()
    tmp = tmp_path_factory.mktemp("stats")
    csv = tmp / "fake.csv"
    df.to_csv(csv, index=False)
    res = stats.run_statistical_tests(str(csv), str(tmp / "tables"))
    return df, truth, res, tmp


def row(tests, test, ref, cond, metric="dice_mean"):
    r = tests[(tests["test_model"] == test) & (tests["ref_model"] == ref) &
              (tests["condition"] == cond) & (tests["metric"] == metric) & (tests["family"] != "drop")]
    assert len(r) == 1, (test, ref, cond, metric)
    return r.iloc[0]


def test_fake_data_shape(run):
    df, truth, _, _ = run
    assert len(df) == 74 * 13 * 8 * 3
    assert df["patient_id"].nunique() == 74
    assert (df[df["condition_id"] == CLEAN]["snr"] == 999.0).all()
    et_empty = df[(df["dice_et"] == 1.0) & (df["hd95_et"] == 373.13)]["patient_id"].nunique()
    assert et_empty == truth["n_empty_et"]


def test_holm_matches_hand_example():
    adj = holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_is_seeded():
    x = np.random.default_rng(0).normal(size=50)
    assert bootstrap_ci(x) == bootstrap_ci(x)


def test_wilcoxon_all_zero_gives_p1():
    assert paired_wilcoxon(np.zeros(10))[1] == 1.0


def test_outputs_written(run):
    _, _, _, tmp = run
    for name in ["statistical_tests", "headline_metrics", "hypotheses", "seed_noise_floor"]:
        assert (tmp / "tables" / f"{name}.csv").exists()


@pytest.mark.parametrize("key,test,ref,cond", [
    ("m4_minus_m0_worst", "M4", "M0", WORST),
    ("m4_minus_m1_worst", "M4", "M1", WORST),
    ("m4_minus_m1_clean", "M4", "M1", CLEAN),
    ("m1_minus_m0_clean", "M1", "M0", CLEAN),
    ("m4_minus_m6_worst", "M4", "M6", WORST),
    ("m5_minus_m6_worst", "M5", "M6", WORST),
])
def test_planted_effect_recovered(run, key, test, ref, cond):
    _, truth, res, _ = run
    r = row(res["statistical_tests"], test, ref, cond)
    planted = truth[key]
    assert r["ci95_low"] - 0.003 <= planted <= r["ci95_high"] + 0.003, (planted, r["median_diff"])
    assert np.sign(r["median_diff"]) == np.sign(planted)
    assert abs(r["median_diff"] - planted) < 0.01


def test_significance_where_planted(run):
    _, _, res, _ = run
    t = res["statistical_tests"]
    assert row(t, "M4", "M0", WORST)["significant"]
    assert row(t, "M4", "M1", WORST)["significant"]
    assert row(t, "M4", "M6", WORST)["significant"]


def test_within_noise_where_planted_below_floor(run):
    _, truth, res, _ = run
    t = res["statistical_tests"]
    # M1 - M0 at clean is planted at 0.004 < 2 x 0.01 floor
    r = row(t, "M1", "M0", CLEAN)
    assert abs(r["median_diff"]) < 2 * truth["seed_noise_floor"]
    assert r["within_training_noise"]
    # the real effects are above the floor
    assert not row(t, "M4", "M0", WORST)["within_training_noise"]
    assert not row(t, "M4", "M1", WORST)["within_training_noise"]


def test_seed_noise_floor_recovered(run):
    _, truth, res, _ = run
    f = res["seed_noise_floor"]
    f = f[(f["metric"] == "dice_mean") & (f["condition"] == CLEAN)]
    assert np.allclose(f["seed_sd"], truth["seed_noise_floor"], atol=0.002)


def test_identical_models_skipped(run):
    _, _, res, _ = run
    t = res["statistical_tests"]
    skipped = t[t["family"] == "skipped"]
    assert set(zip(skipped["test_model"], skipped["ref_model"])) == {("M4", "M2"), ("M4", "M3"), ("M4", "M7")}
    assert skipped["note"].str.contains("identical in code").all()
    assert not ((t["test_model"].isin(["M2", "M3", "M7"])) & (t["family"] != "skipped")).any()


def test_m4_vs_m1_always_present(run):
    _, _, res, _ = run
    t = res["statistical_tests"]
    assert len(t[(t["test_model"] == "M4") & (t["ref_model"] == "M1") & (t["family"] == "primary")]) == 3


def test_holm_applied_within_family(run):
    _, _, res, _ = run
    t = res["statistical_tests"]
    prim = t[t["family"] == "primary"]
    assert (prim["p_holm"] >= prim["raw_p"] - 1e-12).all()
    assert (prim["p_holm"] <= 1.0).all()


def test_hypotheses_recover_planted(run):
    _, truth, res, _ = run
    h = res["hypotheses"].set_index("hypothesis")
    # H1: M4 - M0 at SNR 8 planted ~0.10 > 0.05 margin
    assert h.loc["H1", "verdict"] == "SUPPORTED"
    assert abs(h.loc["H1", "estimate"] - truth["m4_minus_m0_worst"]) < 0.01
    # H2: M0 drops 0.08 vs M4 0.01 -> M4's drop is smaller
    assert h.loc["H2", "verdict"] == "SUPPORTED"
    assert abs(h.loc["H2", "estimate"] - truth["h2_drop_difference"]) < 0.01
    assert h.loc["H2", "p"] < 0.05
    # H3: M5 explains 75 % of M4's gain over M6
    assert abs(h.loc["H3", "estimate"] - truth["h3_fraction"]) < 0.1
    assert h.loc["H3", "verdict"] == "SUPPORTED"


def test_h1_not_supported_when_margin_unreachable(run, monkeypatch):
    _, _, res, tmp = run
    monkeypatch.setitem(stats.H1, "margin", 0.5)
    df, _ = generate_fake_metrics()
    pl = stats.patient_level(df)
    assert stats.eval_h1(pl, stats.seed_noise_floor(df))["verdict"] == "NOT SUPPORTED"


def test_hd95_median_not_mean(run):
    df, _, res, _ = run
    h = res["headline_metrics"]
    r = h[(h["model"] == "M4") & (h["condition"] == CLEAN)].iloc[0]
    # 5 patients carry the 373 mm penalty; median must stay near typical values, mean would not
    assert r["hd95_et_median"] < 20
    assert "hd95_et_q25" in h.columns and "hd95_et_q75" in h.columns
    pl = stats.patient_level(df)
    sub = pl[(pl["model"] == "M4") & (pl["condition_id"] == CLEAN)]
    assert sub["hd95_et"].mean() > 3 * r["hd95_et_median"]


def test_h1_reports_mean_and_median_and_noise_flag_fields(run):
    _, _, res, _ = run
    h1 = res["hypotheses"].set_index("hypothesis").loc["H1"]
    for col in ["estimate", "median", "median_ci95_low", "median_ci95_high"]:
        assert not np.isnan(h1[col])
    assert not h1["within_training_noise"]  # planted ~0.10 effect vs 0.01 floor
    assert "within_training_noise" in res["hypotheses"].columns


def test_h3_aligns_by_patient_id_not_position(run):
    df, _, _, _ = run
    pl = stats.patient_level(df)
    floor = stats.seed_noise_floor(df)
    before = stats.eval_h3(pl, floor)
    shuffled = pl.sample(frac=1.0, random_state=0).reset_index(drop=True)
    after = stats.eval_h3(shuffled, floor)
    assert after["estimate"] == pytest.approx(before["estimate"])
    # drop one patient from M5 only: positional truncation would misalign the rest
    drop_pid = pl["patient_id"].iloc[0]
    partial = pl[~((pl["model"] == "M5") & (pl["patient_id"] == drop_pid))]
    res = stats.eval_h3(partial, floor)
    assert abs(res["estimate"] - before["estimate"]) < 0.1


def test_realistic_regime_h1_not_supported_and_flagged(monkeypatch, tmp_path):
    """Joshua's finding C2/C7: M0 drops only 0.006 and seed SD is 0.015, so nothing beats the noise."""
    from python.analysis import make_fake_metrics as fm
    monkeypatch.setattr(fm, "M0_DROP_WORST", 0.006)
    monkeypatch.setattr(fm, "AUG_DROP_WORST", 0.006)
    monkeypatch.setattr(fm, "SEED_NOISE_SD", 0.015)
    monkeypatch.setattr(fm, "M4_GAIN_OVER_M1", {"clean": 0.005, "worst": 0.005})
    df, truth = fm.generate_fake_metrics()
    assert truth["m0_drop"] == pytest.approx(0.006 * truth["scale_empty_et"])
    csv = tmp_path / "realistic.csv"
    df.to_csv(csv, index=False)
    res = stats.run_statistical_tests(str(csv), str(tmp_path / "tables"))

    floor = res["seed_noise_floor"]
    f = floor[(floor["metric"] == "dice_mean") & (floor["condition"] == WORST) & (floor["model"] == "M0")]
    assert f["seed_sd"].iloc[0] == pytest.approx(0.015, abs=0.002)

    h = res["hypotheses"].set_index("hypothesis")
    assert h.loc["H1", "verdict"] == "NOT SUPPORTED"
    assert h.loc["H1", "within_training_noise"]
    assert abs(h.loc["H1", "estimate"]) < 2 * 0.015
    assert h.loc["H2", "within_training_noise"]

    t = res["statistical_tests"]
    r = row(t, "M4", "M0", WORST)
    assert r["within_training_noise"]
    assert row(t, "M1", "M0", WORST)["within_training_noise"]
