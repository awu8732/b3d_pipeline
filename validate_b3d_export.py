"""
validate_b3d_export.py
======================
Post-processing validation for outputs of process_b3d.py.

Run after process_b3d.py against a single trial output directory:

    python validate_b3d_export.py --trial_dir output/subject10/trial_01 \\
                                  --body      output/subject10/trial_01/body.json

All checks emit PASS / WARN / FAIL with a short explanation.
Exit code 0 if no FAILs, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from b3d_io import read_mot_sto, read_trc   # shared readers; no duplicate parsing


# ── Thresholds ───────────────────────────────────────────────────────────────

# IK
PELVIS_TY_MIN       = 0.80   # m: below this is almost certainly a unit bug
PELVIS_TY_MAX       = 1.20   # m: above this is implausible for walking adults
PELVIS_TX_ABS_MAX   = 5.0    # m: sanity cap; large but not unreasonable for long trials
PELVIS_TRANSLATION_DEG_THRESHOLD = 10.0  # deg: if tx/ty/tz look like degrees, flag it

# GRF
GRF_VERT_MIN_BW     = 0.8    # x body weight: min single-support peak
GRF_VERT_MAX_BW     = 1.5    # x body weight: max single-support peak
COP_RADIUS_MAX      = 0.40   # m: max plausible CoP offset from trial centroid
COP_BELOW_GROUND    = -0.01  # m: CoP py must be >= this

# ID moments
ID_HIP_MAX_NM       = 300.0  # N·m
ID_KNEE_MAX_NM      = 300.0  # N·m
ID_ANKLE_MAX_NM     = 300.0  # N·m
ID_ALL_ZERO_FRAC    = 0.95   # if > this fraction of moment values are 0, flag

# Markers
MARKER_HEEL_Y_MAX   = 0.15   # m: heel markers shouldn't float this high at minimum
MARKER_HEAD_Y_MIN   = 1.20   # m: top-of-head proxies (C7/CLAV) shouldn't be lower

# Timing
TIME_DRIFT_TOL      = 1e-3   # s: max allowed final-timestamp mismatch across files


# ── Result tracking ───────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []  # (status, check_name, message)


def record(status: str, name: str, msg: str) -> None:
    results.append((status, name, msg))
    print(f"  [{status}]  {name}: {msg}")


# ── Column lookup helper ──────────────────────────────────────────────────────

def col_idx(col_names: list[str], name: str) -> int | None:
    try:
        return col_names.index(name)
    except ValueError:
        return None


# ── Check functions ───────────────────────────────────────────────────────────

def check_timing(ik_data, grf_data, id_data, trc_data) -> None:
    """All four files must have the same frame count and matching final timestamp."""
    n_ik  = ik_data.shape[0]
    n_grf = grf_data.shape[0]
    n_id  = id_data.shape[0]
    n_trc = trc_data.shape[0]

    if n_ik == n_grf == n_id == n_trc:
        record(PASS, "frame_count", f"All files have {n_ik} frames")
    else:
        record(FAIL, "frame_count",
               f"Mismatch: ik={n_ik} grf={n_grf} id={n_id} trc={n_trc}")

    # Time column: index 0 for mot/sto, index 1 for trc (col 0 = frame#)
    t_ik  = ik_data[-1, 0]
    t_grf = grf_data[-1, 0]
    t_id  = id_data[-1, 0]
    t_trc = trc_data[-1, 1]
    times = {"ik": t_ik, "grf": t_grf, "id": t_id, "trc": t_trc}
    drift = max(times.values()) - min(times.values())
    if drift < TIME_DRIFT_TOL:
        record(PASS, "time_alignment",
               f"Final timestamps agree within {drift*1000:.2f} ms")
    else:
        record(FAIL, "time_alignment",
               f"Final timestamp drift {drift*1000:.1f} ms: {times}")


def check_ik(ik_cols, ik_data, body: dict) -> None:
    """Pelvis translation plausibility and unit sanity."""
    for name in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
        idx = col_idx(ik_cols, name)
        if idx is None:
            record(FAIL, f"ik_{name}", "Column not found in ik.mot")
            continue
        vals = ik_data[:, idx]

        # Detect if np.degrees() was accidentally applied to translations
        if np.any(np.abs(vals) > PELVIS_TRANSLATION_DEG_THRESHOLD):
            record(FAIL, f"ik_{name}",
                   f"Values look like degrees, not metres "
                   f"(min={vals.min():.3f} max={vals.max():.3f}). "
                   "np.degrees() may have been applied to translation DOFs.")
            continue

        if name == "pelvis_ty":
            lo, hi = vals.min(), vals.max()
            if lo < PELVIS_TY_MIN:
                record(FAIL, "ik_pelvis_ty",
                       f"Min pelvis_ty={lo:.4f} m is below {PELVIS_TY_MIN} m; "
                       "skeleton will appear below ground")
            elif hi > PELVIS_TY_MAX:
                record(WARN, "ik_pelvis_ty",
                       f"Max pelvis_ty={hi:.4f} m exceeds {PELVIS_TY_MAX} m")
            else:
                record(PASS, "ik_pelvis_ty",
                       f"pelvis_ty range [{lo:.4f}, {hi:.4f}] m: plausible")
        else:
            record(PASS, f"ik_{name}",
                   f"range [{vals.min():.4f}, {vals.max():.4f}] m")

    # Check rotational DOFs are in degrees (not radians)
    for name in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"):
        idx = col_idx(ik_cols, name)
        if idx is None:
            continue
        peak = np.abs(ik_data[:, idx]).max()
        if peak < 0.35:  # 0.35 rad ~ 20 deg; walking joints exceed this
            record(WARN, f"ik_{name}_units",
                   f"Peak |{name}|={peak:.4f}: may still be in radians")
        else:
            record(PASS, f"ik_{name}_units",
                   f"Peak |{name}|={peak:.2f} deg: looks like degrees")


def check_grf(grf_cols, grf_data, mass_kg: float) -> None:
    """GRF vertical force peaks and CoP plausibility."""
    g  = 9.81
    bw = mass_kg * g

    r_vy_idx = col_idx(grf_cols, "ground_force_vy")
    l_vy_idx = col_idx(grf_cols, "l_ground_force_vy")
    r_px_idx = col_idx(grf_cols, "ground_force_px")
    r_py_idx = col_idx(grf_cols, "ground_force_py")
    l_py_idx = col_idx(grf_cols, "l_ground_force_py")

    for side, vy_idx in (("right", r_vy_idx), ("left", l_vy_idx)):
        if vy_idx is None:
            record(FAIL, f"grf_{side}_vy", "Column not found")
            continue
        vy      = grf_data[:, vy_idx]
        peak    = vy.max()
        peak_bw = peak / bw
        if peak < 1.0:
            record(WARN, f"grf_{side}_vy",
                   f"Peak vertical GRF = {peak:.1f} N: very low; "
                   "may be swing-only trial or wrong column")
        elif peak_bw < GRF_VERT_MIN_BW:
            record(WARN, f"grf_{side}_vy",
                   f"Peak = {peak:.1f} N ({peak_bw:.2f} BW): below expected "
                   f"{GRF_VERT_MIN_BW} BW for walking single support")
        elif peak_bw > GRF_VERT_MAX_BW:
            record(WARN, f"grf_{side}_vy",
                   f"Peak = {peak:.1f} N ({peak_bw:.2f} BW): above expected "
                   f"{GRF_VERT_MAX_BW} BW; possible force unit issue")
        else:
            record(PASS, f"grf_{side}_vy",
                   f"Peak = {peak:.1f} N ({peak_bw:.2f} BW)")

    # CoP y should be at (or very near) ground level
    for side, py_idx in (("right", r_py_idx), ("left", l_py_idx)):
        if py_idx is None:
            continue
        py     = grf_data[:, py_idx]
        vy_idx = r_vy_idx if side == "right" else l_vy_idx
        loaded = grf_data[:, vy_idx] > 10.0 if vy_idx is not None else np.ones(len(py), bool)
        py_loaded = py[loaded]
        if len(py_loaded) == 0:
            continue
        if py_loaded.min() < COP_BELOW_GROUND:
            record(WARN, f"grf_{side}_cop_y",
                   f"CoP py dips to {py_loaded.min():.4f} m during stance "
                   "(should be ~0)")
        else:
            record(PASS, f"grf_{side}_cop_y",
                   f"CoP py range [{py_loaded.min():.4f}, {py_loaded.max():.4f}] m: "
                   "at ground level")

    # CoP x/z plausibility: should not jump far from trial median
    if r_px_idx is not None:
        px     = grf_data[:, r_px_idx]
        loaded = grf_data[:, r_vy_idx] > 10.0 if r_vy_idx else np.ones(len(px), bool)
        if loaded.any():
            spread = px[loaded].max() - px[loaded].min()
            if spread > COP_RADIUS_MAX * 2:
                record(WARN, "grf_right_cop_x",
                       f"Right CoP x range = {spread:.3f} m: unusually large "
                       "(check left/right foot assignment)")
            else:
                record(PASS, "grf_right_cop_x",
                       f"Right CoP x range = {spread:.3f} m")


def check_id(id_cols, id_data) -> None:
    """Joint moment magnitude and non-zero sanity."""
    checks = [
        ("hip_flexion_r_moment",  ID_HIP_MAX_NM,   "hip"),
        ("knee_angle_r_moment",   ID_KNEE_MAX_NM,  "knee"),
        ("ankle_angle_r_moment",  ID_ANKLE_MAX_NM, "ankle"),
    ]
    for col, limit, label in checks:
        idx = col_idx(id_cols, col)
        if idx is None:
            record(WARN, f"id_{label}", f"Column '{col}' not found")
            continue
        vals = id_data[:, idx]
        peak = np.abs(vals).max()
        if peak == 0.0:
            record(FAIL, f"id_{label}",
                   "All values are zero: pass index may be wrong")
        elif peak > limit:
            record(WARN, f"id_{label}",
                   f"Peak |{col}| = {peak:.1f} N·m exceeds {limit} N·m")
        else:
            record(PASS, f"id_{label}",
                   f"Peak |{col}| = {peak:.1f} N·m")

    # Check overall zero fraction across all moment columns (skip time col)
    moment_data = id_data[:, 1:]
    zero_frac   = np.mean(moment_data == 0.0)
    if zero_frac > ID_ALL_ZERO_FRAC:
        record(FAIL, "id_zero_fraction",
               f"{zero_frac*100:.1f}% of all moment values are zero: "
               "dynamics pass likely not present or wrong pass index")
    else:
        record(PASS, "id_zero_fraction",
               f"{zero_frac*100:.1f}% zero values across all moment columns")


def check_markers(marker_names: list[str], trc_data: np.ndarray, body: dict) -> None:
    """Heel near ground, head markers at plausible height."""
    # trc_data cols: [frame#, time, X0, Y0, Z0, X1, ...]
    # Y col for marker i: 2 + i*3 + 1

    def get_y(name: str) -> Optional[np.ndarray]:
        if name in marker_names:
            i = marker_names.index(name)
            return trc_data[:, 2 + i * 3 + 1]
        return None

    for heel in ("RCAL", "LCAL"):
        y = get_y(heel)
        if y is None:
            record(WARN, f"trc_{heel}", "Marker not found in TRC")
            continue
        y_min = y.min()
        if y_min > MARKER_HEEL_Y_MAX:
            record(WARN, f"trc_{heel}",
                   f"Min Y = {y_min:.4f} m: heel never approaches ground "
                   f"(expected < {MARKER_HEEL_Y_MAX} m)")
        else:
            record(PASS, f"trc_{heel}",
                   f"Min Y = {y_min:.4f} m: reaches near floor")

    for top in ("C7", "CLAV"):
        y = get_y(top)
        if y is None:
            continue
        y_max = y.max()
        if y_max < MARKER_HEAD_Y_MIN:
            record(FAIL, f"trc_{top}",
                   f"Max Y = {y_max:.4f} m: unexpectedly low for a trunk marker "
                   f"(expected > {MARKER_HEAD_Y_MIN} m); markers may be in mm not m")
        else:
            record(PASS, f"trc_{top}", f"Max Y = {y_max:.4f} m")

    all_y            = trc_data[:, 3::3]   # every Y column
    span             = all_y.max() - all_y.min()
    expected_height  = body.get("height_m", None)
    if expected_height:
        ratio = span / expected_height
        if ratio < 0.5:
            record(WARN, "trc_height_span",
                   f"Marker Y span = {span:.3f} m vs expected height "
                   f"{expected_height:.3f} m (ratio={ratio:.2f}): unusually small")
        else:
            record(PASS, "trc_height_span",
                   f"Marker Y span = {span:.3f} m "
                   f"(body height = {expected_height:.3f} m)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate process_b3d.py trial outputs")
    parser.add_argument("--trial_dir", required=True,
                        help="Path to trial output directory (contains ik.mot, grf.mot, etc.)")
    parser.add_argument("--body", required=True,
                        help="Path to body.json for this subject")
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    body_path = Path(args.body)

    print(f"\nValidating: {trial_dir}")
    print("=" * 60)

    with open(body_path) as f:
        body = json.load(f)
    mass_kg = body["mass_kg"]
    print(f"Subject: {body['subject_name']}  "
          f"mass={mass_kg:.1f} kg  height={body['height_m']:.3f} m\n")

    ik_cols,   ik_data  = read_mot_sto(trial_dir / "ik.mot")
    grf_cols,  grf_data = read_mot_sto(trial_dir / "grf.mot")
    id_cols,   id_data  = read_mot_sto(trial_dir / "id_moments.sto")
    trc_names, trc_data = read_trc(trial_dir / "markers.trc")

    print("-- Timing & alignment --")
    check_timing(ik_data, grf_data, id_data, trc_data)

    print("\n-- IK (ik.mot) --")
    check_ik(ik_cols, ik_data, body)

    print("\n-- GRF (grf.mot) --")
    check_grf(grf_cols, grf_data, mass_kg)

    print("\n-- ID moments (id_moments.sto) --")
    check_id(id_cols, id_data)

    print("\n-- Markers (markers.trc) --")
    check_markers(trc_names, trc_data, body)

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n{'='*60}")
    print(f"Summary: {n_pass} passed, {n_warn} warnings, {n_fail} failed")

    if n_fail > 0:
        print("RESULT: FAIL: review errors above before proceeding.")
        sys.exit(1)
    elif n_warn > 0:
        print("RESULT: PASS with warnings: review warnings above.")
    else:
        print("RESULT: PASS")


if __name__ == "__main__":
    main()
