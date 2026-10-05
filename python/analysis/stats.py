"""
Statistics pipeline for the ECTE408 study.

- Per-patient paired Wilcoxon signed-rank (seed-averaged per patient), Holm correction within
  pre-declared families, effect size = median paired difference + bootstrap 95 % CI (seed 42).
- Official H1-H3 as functions (definitions in the HYPOTHESES block below; edit there only).
- Clean-to-degraded drop per patient as its own variable (H2).
- Seed noise floor; differences < 2x floor are flagged "within training noise".
- HD95 reported as median / IQR (the 373.13 mm both-empty penalty, finding C9, wrecks the mean).
- Comparisons involving models identical in code (finding C1) are skipped.

Usage: python -m python.analysis.stats [raw_metrics.csv] [tables_dir]
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAW_METRICS_CSV = "results/raw_metrics.csv"
TABLES_DIR = "tables"

# =============================================================================================
# HYPOTHESES  (edit here once the team records the official wording in docs/DECISIONS.md)
# Current wording = study plan / PAPER_DRAFT, NOT the old stats.py thresholds.
# =============================================================================================
WORST = "snr8_r0.5"
CLEAN = "clean"
ALPHA = 0.05
H1 = {  # M4 beats M0 by >= margin mean Dice at WORST
    "model": "M4", "ref": "M0", "condition": WORST, "metric": "dice_mean", "margin": 0.05,
    "text": "M4 beats M0 by >= 0.05 mean Dice at SNR 8 (r = 0.5)",
}
H2 = {  # M4's clean->WORST drop smaller than M0's (paired Wilcoxon on per-patient drops)
    "model": "M4", "ref": "M0", "metric": "dice_mean", "alpha": ALPHA,
    "text": "M4's clean-to-SNR8 drop in mean Dice is smaller than M0's (paired Wilcoxon, p < 0.05)",
}
H3 = {  # M5 explains >= fraction of M4's gain over M6 at WORST
    "full": "M4", "low_only": "M5", "ref": "M6", "condition": WORST, "metric": "dice_mean", "fraction": 0.5,
    "text": "M5 explains >= 50 % of M4's gain over M6 at SNR 8 (r = 0.5)",
}

# Pre-declared comparison family (test, ref). M4 vs M1 is always included (finding C3).
COMPARISONS = [("M4", "M1"), ("M4", "M0"), ("M1", "M0"), ("M4", "M5"), ("M4", "M6"), ("M5", "M6"),
               ("M4", "M2"), ("M4", "M3"), ("M4", "M7")]
PRIMARY_CONDITIONS = [CLEAN, "snr12_r0.75", WORST]
PRIMARY_METRIC = "dice_mean"
SECONDARY_METRICS = ["dice_wt", "dice_tc", "dice_et"]
DROP_COMPARISONS = [("M4", "M0"), ("M4", "M1")]

# Finding C1: models identical in code. Set SKIP_IDENTICAL = False once Mayank has fixed them.
IDENTICAL_IN_CODE = {"M2": "M1", "M3": "M5", "M7": "M4"}
SKIP_IDENTICAL = True
NOISE_FLOOR_MULTIPLE = 2.0
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42

DICE_COLS = ["dice_wt", "dice_tc", "dice_et", "dice_mean"]
HD_COLS = ["hd95_wt", "hd95_tc", "hd95_et"]
REQUIRED = ["model", "seed", "patient_id", "grade", "condition_id"] + DICE_COLS + HD_COLS


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(x: np.ndarray, stat=np.median, n: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, ci: float = 0.95) -> Tuple[float, float, float]:
    """(point estimate, lo, hi) of `stat` over patients."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boot = stat(x[idx], axis=1)
    a = (1 - ci) / 2
    return float(stat(x)), float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))


def paired_wilcoxon(diffs: np.ndarray) -> Tuple[float, float]:
    """(statistic, two-sided p); p = 1 when all differences are zero."""
    d = np.asarray(diffs, float)
    if len(d) == 0 or np.all(d == 0):
        return 0.0, 1.0
    res = wilcoxon(d, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """Average seeds per (model, patient, condition)."""
    keys = ["model", "patient_id", "grade", "condition_id"]
    cols = [c for c in DICE_COLS + HD_COLS if c in df.columns]
    return df.groupby(keys, as_index=False)[cols].mean()


def _series(pl: pd.DataFrame, model: str, cond: str, metric: str) -> pd.Series:
    sub = pl[(pl["model"] == model) & (pl["condition_id"] == cond)]
    return sub.set_index("patient_id")[metric].sort_index()


def paired_diff(pl: pd.DataFrame, a: str, b: str, cond: str, metric: str) -> np.ndarray:
    sa, sb = _series(pl, a, cond, metric), _series(pl, b, cond, metric)
    common = sa.index.intersection(sb.index)
    return (sa.loc[common] - sb.loc[common]).to_numpy()


def drop_per_patient(pl: pd.DataFrame, model: str, metric: str, cond: str = WORST) -> pd.Series:
    """clean - degraded, per patient (positive = performance lost)."""
    c, d = _series(pl, model, CLEAN, metric), _series(pl, model, cond, metric)
    common = c.index.intersection(d.index)
    return c.loc[common] - d.loc[common]


def seed_noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """SD across seeds of each model's mean Dice, per condition and metric."""
    rows = []
    for (model, cond), g in df.groupby(["model", "condition_id"]):
        for metric in DICE_COLS:
            per_seed = g.groupby("seed")[metric].mean()
            rows.append({"model": model, "condition": cond, "metric": metric, "n_seeds": len(per_seed),
                         "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan})
    return pd.DataFrame(rows)


def _floor(floor_df: pd.DataFrame, model: str, cond: str, metric: str) -> float:
    r = floor_df[(floor_df["model"] == model) & (floor_df["condition"] == cond) & (floor_df["metric"] == metric)]
    return float(r["seed_sd"].iloc[0]) if len(r) else np.nan


def skip_reason(test: str, ref: str, models: List[str]) -> Optional[str]:
    if test not in models or ref not in models:
        return "model not in data"
    if SKIP_IDENTICAL:
        for m in (test, ref):
            if m in IDENTICAL_IN_CODE:
                return f"{m} is identical in code to {IDENTICAL_IN_CODE[m]} (finding C1)"
    return None


# ---------------------------------------------------------------------------------------------
# comparison tests
# ---------------------------------------------------------------------------------------------
def comparison_tests(pl: pd.DataFrame, floor_df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    recs = []
    for test, ref in COMPARISONS:
        reason = skip_reason(test, ref, models)
        if reason:
            recs.append({"family": "skipped", "test_model": test, "ref_model": ref, "note": reason})
            continue
        for cond in PRIMARY_CONDITIONS:
            for metric, family in [(PRIMARY_METRIC, "primary")] + [(m, "secondary") for m in SECONDARY_METRICS]:
                d = paired_diff(pl, test, ref, cond, metric)
                med, lo, hi = bootstrap_ci(d)
                stat, p = paired_wilcoxon(d)
                fl = np.nanmax([_floor(floor_df, test, cond, metric), _floor(floor_df, ref, cond, metric)])
                within = bool(abs(med) < NOISE_FLOOR_MULTIPLE * fl) if not np.isnan(fl) else False
                recs.append({"family": family, "test_model": test, "ref_model": ref, "condition": cond,
                             "metric": metric, "n_patients": len(d), "median_diff": med,
                             "ci95_low": lo, "ci95_high": hi, "wilcoxon_stat": stat, "raw_p": p,
                             "seed_noise_floor": fl, "within_training_noise": within, "note": ""})
    out = pd.DataFrame(recs)
    out["p_holm"] = np.nan
    out["significant"] = False
    for fam in ("primary", "secondary"):  # Holm within each pre-declared family
        m = out["family"] == fam
        if m.any():
            out.loc[m, "p_holm"] = holm(out.loc[m, "raw_p"].to_numpy())
            out.loc[m, "significant"] = out.loc[m, "p_holm"] < ALPHA
    return out


def drop_tests(pl: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Per-patient drop (clean -> WORST) compared between models; Holm within this family."""
    recs = []
    for test, ref in DROP_COMPARISONS:
        if skip_reason(test, ref, models):
            continue
        a, b = drop_per_patient(pl, test, PRIMARY_METRIC), drop_per_patient(pl, ref, PRIMARY_METRIC)
        common = a.index.intersection(b.index)
        d = (b.loc[common] - a.loc[common]).to_numpy()  # positive = `test` loses LESS than `ref`
        med, lo, hi = bootstrap_ci(d)
        stat, p = paired_wilcoxon(d)
        recs.append({"family": "drop", "test_model": test, "ref_model": ref, "condition": WORST,
                     "metric": PRIMARY_METRIC, "n_patients": len(d), "median_diff": med,
                     "ci95_low": lo, "ci95_high": hi, "wilcoxon_stat": stat, "raw_p": p,
                     "note": "drop(ref) - drop(test); positive = test loses less"})
    out = pd.DataFrame(recs)
    if len(out):
        out["p_holm"] = holm(out["raw_p"].to_numpy())
        out["significant"] = out["p_holm"] < ALPHA
    return out


# ---------------------------------------------------------------------------------------------
# official hypotheses
# ---------------------------------------------------------------------------------------------
def _verdict(supported: bool, not_supported: bool) -> str:
    return "SUPPORTED" if supported else ("NOT SUPPORTED" if not_supported else "INCONCLUSIVE")


def eval_h1(pl: pd.DataFrame) -> Dict:
    h = H1
    d = paired_diff(pl, h["model"], h["ref"], h["condition"], h["metric"])
    med, lo, hi = bootstrap_ci(d)
    _, p = paired_wilcoxon(d)
    verdict = _verdict(lo >= h["margin"], hi < h["margin"])  # CI vs the pre-declared margin
    return {"hypothesis": "H1", "text": h["text"], "verdict": verdict, "estimate": med,
            "ci95_low": lo, "ci95_high": hi, "p": p, "threshold": h["margin"],
            "detail": "median paired diff in mean Dice; SUPPORTED if CI lower bound >= margin, "
                      "NOT SUPPORTED if CI upper bound < margin"}


def eval_h2(pl: pd.DataFrame) -> Dict:
    h = H2
    a = drop_per_patient(pl, h["model"], h["metric"])
    b = drop_per_patient(pl, h["ref"], h["metric"])
    common = a.index.intersection(b.index)
    d = (b.loc[common] - a.loc[common]).to_numpy()
    med, lo, hi = bootstrap_ci(d)
    _, p = paired_wilcoxon(d)
    verdict = "SUPPORTED" if (p < h["alpha"] and med > 0) else "NOT SUPPORTED"
    return {"hypothesis": "H2", "text": h["text"], "verdict": verdict, "estimate": med,
            "ci95_low": lo, "ci95_high": hi, "p": p, "threshold": h["alpha"],
            "detail": f"median per-patient (drop {h['ref']} - drop {h['model']}); positive = {h['model']} loses less"}


def eval_h3(pl: pd.DataFrame) -> Dict:
    h = H3
    full = paired_diff(pl, h["full"], h["ref"], h["condition"], h["metric"])
    low = paired_diff(pl, h["low_only"], h["ref"], h["condition"], h["metric"])
    n = min(len(full), len(low))
    full, low = full[:n], low[:n]
    _, p_gain = paired_wilcoxon(full)

    def frac(f, l):
        g = np.mean(f)
        return np.mean(l) / g if g != 0 else np.nan

    est = frac(full, low)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    g = full[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = low[idx].mean(axis=1) / g
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    gain_ok = bool(np.mean(full) > 0 and p_gain < ALPHA)  # a fraction is meaningless without a real gain
    if not gain_ok:
        verdict = "NOT SUPPORTED"
    else:
        verdict = _verdict(lo >= h["fraction"], hi < h["fraction"])
    return {"hypothesis": "H3", "text": h["text"], "verdict": verdict, "estimate": float(est),
            "ci95_low": float(lo), "ci95_high": float(hi), "p": p_gain, "threshold": h["fraction"],
            "detail": f"mean({h['low_only']}-{h['ref']}) / mean({h['full']}-{h['ref']}); p = Wilcoxon on "
                      f"{h['full']} vs {h['ref']} gain; requires a significant positive gain"}


# ---------------------------------------------------------------------------------------------
# headline table
# ---------------------------------------------------------------------------------------------
def headline_table(df: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        for cond in PRIMARY_CONDITIONS:
            sub = df[(df["model"] == model) & (df["condition_id"] == cond)]
            psub = pl[(pl["model"] == model) & (pl["condition_id"] == cond)]
            row = {"model": model, "condition": cond, "n_seeds": sub["seed"].nunique(),
                   "n_patients": psub["patient_id"].nunique()}
            for metric in DICE_COLS:
                s = sub.groupby("seed")[metric].mean()
                row[f"{metric}_mean"] = s.mean()
                row[f"{metric}_seed_sd"] = s.std(ddof=1) if len(s) > 1 else np.nan
            for metric in HD_COLS:
                v = psub[metric].dropna()
                row[f"{metric}_median"] = v.median()
                row[f"{metric}_q25"] = v.quantile(0.25)
                row[f"{metric}_q75"] = v.quantile(0.75)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def run_statistical_tests(raw_metrics_csv: str = RAW_METRICS_CSV, tables_dir: str = TABLES_DIR) -> Dict:
    df = pd.read_csv(raw_metrics_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_metrics_csv} is missing columns: {missing}")
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].unique())
    pl = patient_level(df)
    floor_df = seed_noise_floor(df)

    tests = pd.concat([comparison_tests(pl, floor_df, models), drop_tests(pl, models)], ignore_index=True)
    hyp = pd.DataFrame([eval_h1(pl), eval_h2(pl), eval_h3(pl)])
    headline = headline_table(df, pl)

    tests.to_csv(out_dir / "statistical_tests.csv", index=False)
    headline.to_csv(out_dir / "headline_metrics.csv", index=False)
    hyp.to_csv(out_dir / "hypotheses.csv", index=False)
    floor_df.to_csv(out_dir / "seed_noise_floor.csv", index=False)

    print("\n================ HYPOTHESIS VERDICTS ================")
    for _, r in hyp.iterrows():
        print(f"{r['hypothesis']}: {r['verdict']} - {r['text']}\n    estimate {r['estimate']:+.4f} "
              f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}], p = {r['p']:.3g}")
    print("=====================================================\n")
    return {"statistical_tests": tests, "headline_metrics": headline, "hypotheses": hyp,
            "seed_noise_floor": floor_df}


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else RAW_METRICS_CSV
    tdir = sys.argv[2] if len(sys.argv) > 2 else TABLES_DIR
    run_statistical_tests(csv, tdir)
