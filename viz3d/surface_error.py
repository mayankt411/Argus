"""Signed surface error between a mesh and a reference mask.

Error at each mesh vertex = signed Euclidean distance (mm) to the reference mask surface,
positive outside the reference, negative inside. Reference distance field is
edt(outside) - edt(inside), trilinearly sampled at the vertices.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from viz3d import meshes

DATA_ROOT = Path("/Volumes/PortableSSD/ORVYN/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

PERTURB_MM = 2.0


def signed_distance(mask, spacing):
    """Signed distance (mm) to the mask boundary, positive outside, on the voxel grid."""
    if not mask.any() or mask.all():
        raise ValueError("reference mask must contain both inside and outside voxels")
    return (ndi.distance_transform_edt(~mask, sampling=spacing)
            - ndi.distance_transform_edt(mask, sampling=spacing))


def sample_sdf(sdf, verts_mm, spacing):
    """Trilinearly sample a signed-distance grid at mesh vertices given in mm."""
    idx = (np.asarray(verts_mm) / np.asarray(spacing)).T   # mm -> voxel index coordinates
    return ndi.map_coordinates(sdf, idx, order=1, mode="nearest")


def vertex_errors(verts_mm, reference_mask, spacing):
    """Signed distance of each mesh vertex (mm coordinates) to the reference mask surface."""
    return sample_sdf(signed_distance(reference_mask, spacing), verts_mm, spacing)


def summarize(err):
    return {"mean_signed": float(err.mean()), "mean_abs": float(np.abs(err).mean()),
            "p95_abs": float(np.percentile(np.abs(err), 95)), "max_abs": float(np.abs(err).max()),
            "n_vertices": int(err.size)}


def perturbation_report(mask, spacing, mm=PERTURB_MM):
    """Error of meshes built from the mask itself, dilated by mm, and eroded by mm,
    all measured against the original mask. A perturbation that empties the mask -> None."""
    sdf = signed_distance(mask, spacing)
    out = {}
    for name, m in (("self", mask), (f"dilate_{mm:g}mm", meshes.dilate_mask(mask, spacing, mm)),
                    (f"erode_{mm:g}mm", meshes.erode_mask(mask, spacing, mm))):
        mesh = meshes.mask_to_mesh(m, spacing)
        if mesh is None:
            out[name] = None
            continue
        out[name] = summarize(sample_sdf(sdf, mesh[0], spacing))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", default=["002"])
    ap.add_argument("--region", default="WT", choices=list(meshes.REGIONS))
    args = ap.parse_args()
    for c in args.cases:
        seg, sp, _ = meshes.load_seg(DATA_ROOT, c)
        print(f"{meshes.case_name(c)}  region={args.region}")
        for k, v in perturbation_report(meshes.region_mask(seg, args.region), sp).items():
            print(f"  {k:12s}", "empty mesh" if v is None else
                  f"signed {v['mean_signed']:+.3f}  abs {v['mean_abs']:.3f}  "
                  f"p95 {v['p95_abs']:.3f}  max {v['max_abs']:.3f}  (n={v['n_vertices']})")


if __name__ == "__main__":
    main()
