"""Segmentation -> surface mesh utilities for BraTS 2020 training cases.

Meshes are in millimetres, in the scan's own voxel grid: vertex = voxel_index * spacing,
so voxel centre (i, j, k) sits at (i, j, k) * spacing. No affine is applied.
"""
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

DATA_ROOT = Path("/Volumes/PortableSSD/ORVYN/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

SIGMA_MM = 0.5   # Gaussian pre-smoothing of the binary mask before marching cubes
LEVEL = 0.5

CASE_RE = re.compile(r"^BraTS20_Training_\d{3}$")

# BraTS labels: 1 = necrotic/non-enhancing core, 2 = peritumoral edema, 4 = enhancing tumour.
LABELS = {"NCR": (1,), "ED": (2,), "ET": (4,)}
REGIONS = {
    "WT": (1, 2, 4),   # whole tumour
    "TC": (1, 4),      # tumour core
    "ET": (4,),
    "NCR": (1,),
    "ED": (2,),
}


def case_name(case):
    """'2', '002' or 'BraTS20_Training_002' -> 'BraTS20_Training_002'."""
    case = str(case)
    name = case if case.startswith("BraTS20_") else f"BraTS20_Training_{int(case):03d}"
    if not CASE_RE.match(name):
        raise ValueError(f"bad BraTS case id: {case!r}")
    return name


def list_cases(data_root):
    """Case directory names under data_root, skipping macOS '._' junk."""
    return sorted(p.name for p in Path(data_root).iterdir()
                  if p.is_dir() and not p.name.startswith("._") and CASE_RE.match(p.name))


def seg_path(data_root, case):
    name = case_name(case)
    path = Path(data_root) / name / f"{name}_seg.nii"
    if path.name.startswith("._"):
        raise ValueError(f"refusing macOS junk file: {path}")
    return path


def load_seg(data_root, case):
    """Return (seg uint8 array, spacing (3,) float64 mm, axis codes e.g. ('L','P','S'))."""
    img = nib.load(seg_path(data_root, case))
    seg = np.asarray(img.dataobj).astype(np.uint8)
    bad = set(np.unique(seg).tolist()) - {0, 1, 2, 4}
    if bad:
        raise ValueError(f"{case}: unexpected labels {sorted(bad)}")
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    return seg, spacing, nib.aff2axcodes(img.affine)


def region_mask(seg, region):
    """Boolean mask for a region name in REGIONS."""
    return np.isin(seg, REGIONS[region])


def mask_to_mesh(mask, spacing, sigma_mm=SIGMA_MM, level=LEVEL):
    """Smoothed marching-cubes mesh of a boolean mask -> (verts_mm, faces), or None if empty."""
    if not mask.any():
        return None
    spacing = np.asarray(spacing, dtype=np.float64)
    vol = mask.astype(np.float32)
    if sigma_mm:
        vol = ndi.gaussian_filter(vol, sigma_mm / spacing)
    vol = np.pad(vol, 1)   # zero border so surfaces touching the volume edge stay closed
    if not vol.min() < level < vol.max():
        return None
    verts, faces, _, _ = measure.marching_cubes(vol, level, spacing=tuple(spacing))
    return verts - spacing, faces   # undo the 1-voxel pad


def dilate_mask(mask, spacing, mm):
    """Euclidean dilation by `mm` (anisotropy-aware)."""
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= mm


def erode_mask(mask, spacing, mm):
    """Euclidean erosion by `mm` (anisotropy-aware); may return an empty mask."""
    return ndi.distance_transform_edt(mask, sampling=spacing) > mm


if __name__ == "__main__":
    import sys
    for c in sys.argv[1:] or ["002"]:
        seg, sp, codes = load_seg(DATA_ROOT, c)
        print(case_name(c), seg.shape, sp, codes)
        for r in REGIONS:
            m = mask_to_mesh(region_mask(seg, r), sp)
            print(f"  {r:3s}", "empty" if m is None else f"{len(m[0])} verts, {len(m[1])} faces")
