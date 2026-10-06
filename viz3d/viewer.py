"""Truth-only interactive 3D viewer: one self-contained HTML per case, built from the ground-truth
segmentation (no predictions). Click legend entries to toggle labels."""
import argparse
from pathlib import Path

import plotly.graph_objects as go

from viz3d import meshes

DATA_ROOT = Path("/Volumes/PortableSSD/ORVYN/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "interactive"

DEFAULT_CASES = ["008", "167", "236", "292", "281"]

# key -> (mask region, colour, opacity, visible). WT, TC and ET are shown by default; ED and NCR are legend toggles.
STYLE = {
    "WT":  ("WT", "#1f77b4", 0.25, True),
    "TC":  ("TC", "#ff7f0e", 0.35, True),
    "ET":  ("ET", "#d62728", 0.90, True),
    "ED":  ("ED", "#e6c229", 0.25, "legendonly"),
    "NCR": ("NCR", "#7b3fa0", 0.60, "legendonly"),
}
FULL_NAMES = {"WT": "Whole tumour (1+2+4)", "TC": "Tumour core (1+4)", "ED": "Edema (2)", "NCR": "Necrotic/non-enhancing (1)", "ET": "Enhancing (4)"}


def build_figure(seg, spacing, axcodes, title):
    voxel_ml = float(spacing.prod()) / 1000.0
    fig = go.Figure()
    volumes = []
    for key, (region, colour, opacity, visible) in STYLE.items():
        mask = meshes.region_mask(seg, region)
        mesh = meshes.mask_to_mesh(mask, spacing, sigma_mm=meshes.SIGMA_MM, level=meshes.LEVEL)   # 0.5 / 0.5 for every layer
        if mesh is None:   # e.g. case 281 has no enhancing tumour
            continue
        v, f = mesh
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=colour, opacity=opacity, name=f"{FULL_NAMES[key]} — {mask.sum() * voxel_ml:.1f} mL",
            showlegend=True, visible=visible, flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
            hovertemplate=f"{FULL_NAMES[key]}<br>x %{{x:.1f}} y %{{y:.1f}} z %{{z:.1f}} mm<extra></extra>"))
        if key in ("WT", "TC", "ET"):
            volumes.append(f"{key} {mask.sum() * voxel_ml:.1f} mL")
    axis = lambda a, c: dict(title=f"{a} (→ {c}) mm", backgroundcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=f"{title} — ground truth<br><sup>{' · '.join(volumes)}</sup>"),
        scene=dict(aspectmode="data", xaxis=axis("x", axcodes[0]), yaxis=axis("y", axcodes[1]),
                   zaxis=axis("z", axcodes[2])),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
    return fig


def build_viewer(data_root, case, out_dir):
    name = meshes.case_name(case)
    seg, spacing, axcodes = meshes.load_seg(data_root, name)
    fig = build_figure(seg, spacing, axcodes, name)
    out = Path(out_dir) / f"{name}_truth.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)   # embedded JS: works offline
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    for c in args.cases:
        out = build_viewer(DATA_ROOT, c, args.out_dir)
        print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
