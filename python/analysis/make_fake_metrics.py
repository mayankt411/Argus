"""
Synthetic raw_metrics.csv with KNOWN planted effects, for testing stats.py.

Same columns as python/evaluate.py. 74 patients x 13 conditions x 8 models x 3 seeds.
The planted effects are returned by `generate_fake_metrics` (and written next to the CSV as
`*_truth.json`) so tests can check that the pipeline recovers them.

Never writes to results/raw_metrics.csv.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

OUT_CSV = "data/cache/raw_metrics_fake.csv"  # gitignored (data/cache/)
SEED = 42

CONDITIONS = [("clean", None, 1.0)] + [
    (f"snr{int(s)}_r{r}", float(s), r) for s in (30, 20, 12, 8) for r in (1.0, 0.75, 0.5)
]
MODELS = [f"M{i}" for i in range(8)]
SEEDS = [0, 1, 2]
N_PATIENTS, N_LGG, N_LGG_NO_ET = 74, 15, 5

# ----- planted effects (mean-Dice units; applied equally to WT/TC/ET, see `_effect_scale`) -----
BASE_REGION_DICE = {"wt": 0.78, "tc": 0.72, "et": 0.66}
PATIENT_SD = 0.05            # shared across models -> paired design
PATIENT_MODEL_SD = 0.015     # patient x model interaction (fixed across seeds/conditions)
ROW_NOISE_SD = 0.01
SEED_NOISE_SD = 0.01         # exact sample SD of each model's three seed offsets
M0_DROP_WORST = 0.08         # M0 clean -> snr8_r0.5
AUG_DROP_WORST = 0.01        # augmentation-trained models (M1-M7) clean -> snr8_r0.5
M4_GAIN_OVER_M1 = {"clean": 0.01, "worst": 0.03}  # M4 - M1 at clean / at snr8_r0.5
M5_GAIN_OVER_M6 = 0.03       # M4 - M6 = 0.04 -> M5 explains 75 % of M4's gain over M6
M4_GAIN_OVER_M6 = 0.04
M1_GAIN_OVER_M0_CLEAN = 0.004  # below 2 x seed floor (0.02): "within training noise"
# models identical in code (finding C1): same planted effect, different seed noise
ALIASES = {"M2": "M1", "M3": "M5", "M7": "M4"}


def _severity(snr, r) -> float:
    """0 at clean / (snr30, r1), 1 at (snr8, r0.5)."""
    if snr is None:
        return 0.0
    s_snr = (1 / snr - 1 / 30) / (1 / 8 - 1 / 30)
    s_r = (1 - r) / 0.5
    return 0.6 * s_snr + 0.4 * s_r


def _model_level(model: str, sev: float) -> float:
    """Planted mean-Dice level of a (canonical) model at severity `sev`."""
    model = ALIASES.get(model, model)
    if model == "M0":
        return -M0_DROP_WORST * sev
    level = -AUG_DROP_WORST * sev
    # M1 baseline for the augmented family; other models offset relative to it
    if model == "M1":
        return level + M1_GAIN_OVER_M0_CLEAN
    if model == "M4":
        gain = M4_GAIN_OVER_M1["clean"] + (M4_GAIN_OVER_M1["worst"] - M4_GAIN_OVER_M1["clean"]) * sev
        return level + M1_GAIN_OVER_M0_CLEAN + gain
    m4 = _model_level("M4", sev)
    if model == "M6":
        return m4 - M4_GAIN_OVER_M6
    if model == "M5":
        return m4 - M4_GAIN_OVER_M6 + M5_GAIN_OVER_M6
    raise ValueError(model)


def _effect_scale() -> float:
    """Planted effects hit WT/TC/ET equally except for the 5 empty-ET patients (ET fixed at 1.0)."""
    n_nonempty = N_PATIENTS - N_LGG_NO_ET
    return (n_nonempty + N_LGG_NO_ET * 2 / 3) / N_PATIENTS


def planted_truth() -> Dict:
    """Expected population effects on mean Dice, in the units the pipeline reports."""
    k = _effect_scale()
    lvl = lambda m, c: _model_level(m, _severity(*c)) * k
    worst, clean = (8.0, 0.5), (None, 1.0)
    return {
        "scale_empty_et": k,
        "seed_noise_floor": SEED_NOISE_SD,
        "m4_minus_m0_worst": lvl("M4", worst) - lvl("M0", worst),
        "m4_minus_m1_worst": lvl("M4", worst) - lvl("M1", worst),
        "m4_minus_m1_clean": lvl("M4", clean) - lvl("M1", clean),
        "m1_minus_m0_clean": lvl("M1", clean) - lvl("M0", clean),
        "m0_drop": lvl("M0", clean) - lvl("M0", worst),
        "m4_drop": lvl("M4", clean) - lvl("M4", worst),
        "h2_drop_difference": (lvl("M0", clean) - lvl("M0", worst)) - (lvl("M4", clean) - lvl("M4", worst)),
        "m4_minus_m6_worst": M4_GAIN_OVER_M6 * k,
        "m5_minus_m6_worst": M5_GAIN_OVER_M6 * k,
        "h3_fraction": M5_GAIN_OVER_M6 / M4_GAIN_OVER_M6,
        "n_empty_et": N_LGG_NO_ET,
    }


def generate_fake_metrics(seed: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)
    pids = [f"BraTS20_Training_{i:03d}" for i in range(1, N_PATIENTS + 1)]
    grade = np.array(["HGG"] * (N_PATIENTS - N_LGG) + ["LGG"] * N_LGG)
    empty_et = np.zeros(N_PATIENTS, bool)
    empty_et[N_PATIENTS - N_LGG: N_PATIENTS - N_LGG + N_LGG_NO_ET] = True  # the first 5 LGG

    patient_eff = rng.normal(0, PATIENT_SD, N_PATIENTS)
    pm_eff = {m: rng.normal(0, PATIENT_MODEL_SD, N_PATIENTS) for m in MODELS}
    seed_off = {}
    for m in MODELS:
        off = np.array([-1.0, 0.0, 1.0]) * SEED_NOISE_SD  # sample SD (ddof=1) == SEED_NOISE_SD
        seed_off[m] = rng.permutation(off)

    vol_gt = {k: np.exp(rng.normal(np.log(v), 0.6, N_PATIENTS)) for k, v in
              {"wt": 80.0, "tc": 28.0, "et": 12.0}.items()}
    for i in np.where(empty_et)[0]:
        vol_gt["et"][i] = 0.0

    rows = []
    for m in MODELS:
        for si, s in enumerate(SEEDS):
            for cid, snr, r in CONDITIONS:
                sev = _severity(snr, r)
                level = _model_level(m, sev)
                shared = level + patient_eff + pm_eff[m] + seed_off[m][si]
                dice = {}
                for reg, base in BASE_REGION_DICE.items():
                    d = base + shared + rng.normal(0, ROW_NOISE_SD, N_PATIENTS)
                    dice[reg] = np.clip(d, 0.0, 1.0)
                dice["et"] = np.where(empty_et, 1.0, dice["et"])  # BraTS: both empty -> 1

                # HD95: heavy-tailed, worse with severity / lower Dice; 373.13 when both empty (finding C9)
                hd = {}
                for reg in ("wt", "tc", "et"):
                    h = np.exp(rng.normal(np.log(4.0 + 6.0 * sev), 0.7, N_PATIENTS)) * (1.2 - dice[reg])
                    hd[reg] = h
                hd["et"] = np.where(empty_et, 373.13, hd["et"])

                vol_pred = {}
                for reg in ("wt", "tc", "et"):
                    vol_pred[reg] = np.maximum(vol_gt[reg] * (1 + rng.normal(0, 0.15 + 0.2 * (1 - dice[reg]), N_PATIENTS)), 0)
                    vol_pred[reg] = np.where(vol_gt[reg] == 0, 0.0, vol_pred[reg])
                err = np.abs(vol_pred["wt"] - vol_gt["wt"])
                core_gt = np.where(vol_gt["tc"] > 0, (vol_gt["wt"] - vol_gt["tc"]) / vol_gt["tc"], 0.0)
                core_pr = np.where(vol_pred["tc"] > 0, (vol_pred["wt"] - vol_pred["tc"]) / vol_pred["tc"], 0.0)

                for i in range(N_PATIENTS):
                    rows.append({
                        "model": m, "seed": s, "patient_id": pids[i], "grade": grade[i],
                        "condition_id": cid, "snr": 999.0 if snr is None else snr, "r": r,
                        "dice_wt": dice["wt"][i], "dice_tc": dice["tc"][i], "dice_et": dice["et"][i],
                        "dice_mean": (dice["wt"][i] + dice["tc"][i] + dice["et"][i]) / 3.0,
                        "hd95_wt": hd["wt"][i], "hd95_tc": hd["tc"][i], "hd95_et": hd["et"][i],
                        "vol_pred_wt_cm3": vol_pred["wt"][i], "vol_gt_wt_cm3": vol_gt["wt"][i],
                        "vol_abs_err_wt_cm3": err[i], "vol_rel_err_wt": err[i] / vol_gt["wt"][i],
                        "vol_pred_tc_cm3": vol_pred["tc"][i], "vol_gt_tc_cm3": vol_gt["tc"][i],
                        "vol_pred_et_cm3": vol_pred["et"][i], "vol_gt_et_cm3": vol_gt["et"][i],
                        "edema_core_ratio_pred": core_pr[i], "edema_core_ratio_gt": core_gt[i],
                        "edema_core_ratio_abs_err": abs(core_pr[i] - core_gt[i]),
                    })
    return pd.DataFrame(rows), planted_truth()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    out = Path(args.out)
    if out.name == "raw_metrics.csv":
        raise SystemExit("Refusing to write fake data to a file named raw_metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df, truth = generate_fake_metrics()
    df.to_csv(out, index=False)
    out.with_name(out.stem + "_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(df)} fake rows to {out}")


if __name__ == "__main__":
    main()
