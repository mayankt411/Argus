from pathlib import Path

import numpy as np
import pytest

from viz3d import meshes, surface_error, viewer

DATA_ROOT = Path("/Volumes/PortableSSD/ORVYN/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

needs_data = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="BraTS data root not mounted")

# Joshua's values for BraTS20_Training_002 (whole tumour), tolerance 0.1 mm.
JOSHUA_002 = {"self_signed": -0.03, "self_abs": 0.20, "dilate_2mm": +2.04, "erode_2mm": -2.05}
TOL_MM = 0.1


def sphere(shape=(60, 60, 60), centre=(30, 30, 30), radius=15.0, spacing=(1.0, 1.0, 1.0)):
    g = np.indices(shape).astype(float)
    d2 = sum(((g[a] - centre[a]) * spacing[a]) ** 2 for a in range(3))
    return d2 <= radius ** 2


# ---- synthetic checks (no data needed) -------------------------------------------------------

def test_list_cases_ignores_macos_junk(tmp_path):
    for n in ("BraTS20_Training_001", "BraTS20_Training_002", "._BraTS20_Training_003"):
        (tmp_path / n).mkdir()
    (tmp_path / "._BraTS20_Training_004").write_bytes(b"\0")   # junk file, not a dir
    (tmp_path / "name_mapping.csv").write_text("x")
    assert meshes.list_cases(tmp_path) == ["BraTS20_Training_001", "BraTS20_Training_002"]


@pytest.mark.parametrize("bad", ["../etc", "BraTS20_Training_2", "._BraTS20_Training_002", "abc"])
def test_case_name_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        meshes.case_name(bad)


def test_case_name_normalises():
    assert meshes.case_name("2") == meshes.case_name("002") == "BraTS20_Training_002"
    assert meshes.seg_path("/x", "002").name == "BraTS20_Training_002_seg.nii"


def test_empty_mask_gives_no_mesh():
    assert meshes.mask_to_mesh(np.zeros((8, 8, 8), bool), (1, 1, 1)) is None


def test_sphere_self_error_small_and_closed_at_border():
    m = sphere()
    verts, faces = meshes.mask_to_mesh(m, (1, 1, 1))
    s = surface_error.summarize(surface_error.vertex_errors(verts, m, (1, 1, 1)))
    assert abs(s["mean_signed"]) < 0.1 and s["mean_abs"] < 0.3
    # a mask touching the volume edge must still produce a mesh (zero padding closes it)
    edge = np.ones((10, 10, 10), bool)
    assert meshes.mask_to_mesh(edge, (1, 1, 1)) is not None


def test_anisotropic_spacing_is_in_mm():
    sp = (1.0, 1.0, 3.0)
    m = sphere(shape=(60, 60, 30), centre=(30, 30, 15), radius=15.0, spacing=sp)
    verts, _ = meshes.mask_to_mesh(m, sp)
    ext = verts.max(0) - verts.min(0)
    assert np.allclose(ext, 30, atol=3.5)   # ~diameter in mm on every axis, incl. the 3 mm axis
    rep = surface_error.perturbation_report(m, sp, 2.0)
    assert abs(rep["dilate_2mm"]["mean_signed"] - 2.0) < 0.35
    assert abs(rep["erode_2mm"]["mean_signed"] + 2.0) < 0.35


def test_perturbation_signs_and_empty_erosion():
    m = sphere(radius=15.0)
    rep = surface_error.perturbation_report(m, (1, 1, 1), 2.0)
    assert rep["dilate_2mm"]["mean_signed"] > 1.5 > -1.5 > rep["erode_2mm"]["mean_signed"]
    tiny = sphere(radius=1.5)   # nothing survives a 2 mm erosion
    assert surface_error.perturbation_report(tiny, (1, 1, 1), 2.0)["erode_2mm"] is None


def test_signed_distance_needs_both_sides():
    with pytest.raises(ValueError):
        surface_error.signed_distance(np.zeros((4, 4, 4), bool), (1, 1, 1))


# ---- real data -------------------------------------------------------------------------------

@needs_data
def test_matches_joshua_002():
    seg, sp, _ = meshes.load_seg(DATA_ROOT, "002")
    assert tuple(sp) == (1.0, 1.0, 1.0) and seg.shape == (240, 240, 155)
    rep = surface_error.perturbation_report(meshes.region_mask(seg, "WT"), sp, 2.0)
    got = {"self_signed": rep["self"]["mean_signed"], "self_abs": rep["self"]["mean_abs"],
           "dilate_2mm": rep["dilate_2mm"]["mean_signed"], "erode_2mm": rep["erode_2mm"]["mean_signed"]}
    for k, want in JOSHUA_002.items():
        assert abs(got[k] - want) <= TOL_MM, f"{k}: got {got[k]:+.3f}, want {want:+.2f}"


@needs_data
def test_only_real_files_listed():
    cases = meshes.list_cases(DATA_ROOT)
    assert len(cases) == 369 and not any(c.startswith("._") for c in cases)


@needs_data
@pytest.mark.parametrize("case", ["281", "008"])
def test_viewer_truth_only(case, tmp_path):
    out = viewer.build_viewer(DATA_ROOT, case, tmp_path)
    html = out.read_text()
    assert out.name == f"BraTS20_Training_{case}_truth.html"
    assert "Mesh3d" in html or "mesh3d" in html
    seg, sp, _ = meshes.load_seg(DATA_ROOT, case)
    fig = viewer.build_figure(seg, sp, ("L", "P", "S"), case)
    names = [t.name for t in fig.data]
    assert not any("pred" in n.lower() for n in names)
    has_et = bool((seg == 4).any())
    assert any(n.startswith("Enhancing") for n in names) == has_et   # 281 has no label 4
    for t in fig.data:
        assert np.isfinite(t.x).all() and np.isfinite(t.y).all() and np.isfinite(t.z).all()


@needs_data
def test_layer_defaults_and_mesh_params():
    assert meshes.SIGMA_MM == 0.5 and meshes.LEVEL == 0.5
    assert {k: v[1:] for k, v in viewer.STYLE.items()} == {
        "WT": ("#1f77b4", 0.25, True), "TC": ("#ff7f0e", 0.35, True), "ET": ("#d62728", 0.90, True),
        "ED": ("#e6c229", 0.25, "legendonly"), "NCR": ("#7b3fa0", 0.60, "legendonly")}
    seg, sp, _ = meshes.load_seg(DATA_ROOT, "008")
    assert (meshes.region_mask(seg, "TC") == ((seg == 1) | (seg == 4))).all()
    fig = viewer.build_figure(seg, sp, ("L", "P", "S"), "008")
    by = {t.name.split(" — ")[0]: t for t in fig.data}
    for key, label in (("WT", "Whole tumour (1+2+4)"), ("TC", "Tumour core (1+4)"), ("ET", "Enhancing (4)")):
        want, _ = meshes.mask_to_mesh(meshes.region_mask(seg, key), sp, sigma_mm=0.5, level=0.5)
        assert np.allclose(np.column_stack([by[label].x, by[label].y, by[label].z]), want)
