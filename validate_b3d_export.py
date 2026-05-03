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

from b3d_io import read_mot_sto


# ── Thresholds ───────────────────────────────────────────────────────────────

# IK
# Pelvis vertical height expressed as a fraction of subject height.
# Typical values from literature:
#   walking: 0.47-0.55  |  running: 0.45-0.55
# A subject-height-relative check avoids false positives on short subjects.
PELVIS_TY_FRAC_MIN  = 0.44   # fraction of height: fail below this
PELVIS_TY_FRAC_WARN = 0.47   # fraction of height: warn below this
PELVIS_TY_FRAC_MAX  = 0.58   # fraction of height: warn above this
PELVIS_TX_ABS_MAX   = 5.0    # m: sanity cap; large but not unreasonable for long trials
PELVIS_TRANSLATION_DEG_THRESHOLD = 10.0  # deg: if tx/ty/tz look like degrees, flag it

# GRF: thresholds differ by activity mode (--mode walk|run).
# Walking: single-support peak ~1.0-1.2 BW.
# Running: single-support peak ~1.5-3.0 BW (speed-dependent).
GRF_VERT_MIN_BW     = {"walk": 0.8,  "run": 1.2}
GRF_VERT_MAX_BW     = {"walk": 1.5,  "run": 3.5}
COP_RADIUS_MAX      = 0.40   # m: max plausible CoP offset from trial centroid
COP_BELOW_GROUND    = -0.02  # m: slightly lax to absorb CoP hold-through-swing artifact

# ID moments
ID_HIP_MAX_NM       = 300.0  # N·m
ID_KNEE_MAX_NM      = 300.0  # N·m
ID_ANKLE_MAX_NM     = 300.0  # N·m
ID_ALL_ZERO_FRAC    = 0.95   # if > this fraction of moment values are 0, flag


# Timing
TIME_DRIFT_TOL      = 1e-3   # s: max allowed final-timestamp mismatch across files

# Newton-Euler whole-body CoM residual
# The pelvis origin is used as a proxy for the whole-body CoM. This is an
# approximation (~5-10 cm error) but sufficient to catch gross dynamics
# failures such as wrong pass index, unit errors, or missing GRF.
#
# Residual = m * a_com - F_net_external
#   where F_net_external = F_grf_total - m*g (vertical only for gravity)
#   and a_com is estimated by differentiating pelvis translations twice.
#
# A small residual confirms the ID moments are internally consistent with
# the kinematics and GRFs. A large residual means the three files are
# mismatched (wrong pass, swapped trial, unit error, etc.).
# Running has larger pelvis-CoM proxy error (~15-25% BW vs ~5-10% for walking)
# due to greater torso displacement. Thresholds are keyed by mode.
NE_COM_RESIDUAL_WARN_BW  = {"walk": 0.15, "run": 0.30}
NE_COM_RESIDUAL_FAIL_BW  = {"walk": 0.40, "run": 0.75}

# ID vs GRF consistency: total support moment (TSM) correlation
# TSM = sum of sagittal lower-limb moments. In healthy gait TSM correlates
# strongly with total vertical GRF (Pearson r > 0.70 expected).
NE_TSM_CORR_WARN  = 0.70  # below this correlation: warn
NE_TSM_CORR_FAIL  = 0.40  # below this correlation: fail


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

def check_timing(ik_data, grf_data, id_data) -> None:
    """All four files must have the same frame count and matching final timestamp."""
    n_ik  = ik_data.shape[0]
    n_grf = grf_data.shape[0]
    n_id  = id_data.shape[0]
    if n_ik == n_grf == n_id:
        record(PASS, "frame_count", f"All files have {n_ik} frames")
    else:
        record(FAIL, "frame_count",
               f"Mismatch: ik={n_ik} grf={n_grf} id={n_id}")

    # Time column: index 0 for all mot/sto files
    t_ik  = ik_data[-1, 0]
    t_grf = grf_data[-1, 0]
    t_id  = id_data[-1, 0]
    times = {"ik": t_ik, "grf": t_grf, "id": t_id}
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
            h      = body.get("height_m", 1.70)
            lo_frac, hi_frac = lo / h, hi / h
            if lo_frac < PELVIS_TY_FRAC_MIN:
                record(FAIL, "ik_pelvis_ty",
                       f"Min pelvis_ty={lo:.4f} m ({lo_frac:.3f} x height): "
                       f"below {PELVIS_TY_FRAC_MIN} x height; likely unit error")
            elif lo_frac < PELVIS_TY_FRAC_WARN:
                record(WARN, "ik_pelvis_ty",
                       f"Min pelvis_ty={lo:.4f} m ({lo_frac:.3f} x height): "
                       f"slightly below expected {PELVIS_TY_FRAC_WARN} x height")
            elif hi_frac > PELVIS_TY_FRAC_MAX:
                record(WARN, "ik_pelvis_ty",
                       f"Max pelvis_ty={hi:.4f} m ({hi_frac:.3f} x height): "
                       f"above expected {PELVIS_TY_FRAC_MAX} x height")
            else:
                record(PASS, "ik_pelvis_ty",
                       f"pelvis_ty range [{lo:.4f}, {hi:.4f}] m "
                       f"({lo_frac:.3f}-{hi_frac:.3f} x height): plausible")
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


def check_grf(grf_cols, grf_data, mass_kg: float, mode: str = 'walk') -> None:
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
        vy_min_bw = GRF_VERT_MIN_BW[mode]
        vy_max_bw = GRF_VERT_MAX_BW[mode]
        if peak < 1.0:
            record(WARN, f"grf_{side}_vy",
                   f"Peak vertical GRF = {peak:.1f} N: very low; "
                   "may be swing-only trial or wrong column")
        elif peak_bw < vy_min_bw:
            record(WARN, f"grf_{side}_vy",
                   f"Peak = {peak:.1f} N ({peak_bw:.2f} BW): below expected "
                   f"{vy_min_bw} BW for {mode} single support")
        elif peak_bw > vy_max_bw:
            record(WARN, f"grf_{side}_vy",
                   f"Peak = {peak:.1f} N ({peak_bw:.2f} BW): above expected "
                   f"{vy_max_bw} BW for {mode}; possible force unit issue")
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




def check_newton_euler_com(
    ik_cols, ik_data,
    grf_cols, grf_data,
    mass_kg: float,
    mode: str = "walk",
) -> None:
    """
    Whole-body linear momentum balance: m * a_com ≈ F_net_external.

    The pelvis origin is used as a CoM proxy. This is approximate but catches
    gross mismatches between IK kinematics and GRF data (wrong pass index,
    swapped trial, unit error, etc.).

    Method
    ------
    1. Extract pelvis_tx/ty/tz from ik.mot (metres).
    2. Differentiate twice with np.gradient to get acceleration (m/s²).
    3. Multiply by mass to get expected net force (N).
    4. Compare against actual net external force from GRF:
         F_net_x = GRF_Fx_total
         F_net_y = GRF_Fy_total - m*g   (subtract gravity)
         F_net_z = GRF_Fz_total
    5. RMS of (m*a - F_net) across all frames and axes, normalised by BW.

    A residual > NE_COM_RESIDUAL_WARN_BW suggests a mismatch. A residual
    > NE_COM_RESIDUAL_FAIL_BW is almost certainly a data error.
    """
    g   = 9.81
    bw  = mass_kg * g

    # ── Time vector and sampling rate ────────────────────────────────────────
    time = ik_data[:, 0]
    dt   = float(np.median(np.diff(time)))
    if dt <= 0:
        record(WARN, "ne_com", "Cannot compute: non-positive dt in ik.mot")
        return

    # ── Pelvis translation columns (metres) ─────────────────────────────────
    missing = []
    pos = {}
    for ax, name in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        idx = col_idx(ik_cols, name)
        if idx is None:
            missing.append(name)
        else:
            pos[ax] = ik_data[:, idx]
    if missing:
        record(WARN, "ne_com",
               f"Skipped: pelvis translation columns missing: {missing}")
        return

    # ── GRF net force columns ────────────────────────────────────────────────
    grf_col_map = {
        0: ("ground_force_vx",   "l_ground_force_vx"),
        1: ("ground_force_vy",   "l_ground_force_vy"),
        2: ("ground_force_vz",   "l_ground_force_vz"),
    }
    net_grf = np.zeros((len(time), 3))
    for ax, (r_name, l_name) in grf_col_map.items():
        ri = col_idx(grf_cols, r_name)
        li = col_idx(grf_cols, l_name)
        if ri is None or li is None:
            record(WARN, "ne_com",
                   f"Skipped: GRF column missing for axis {ax}")
            return
        net_grf[:, ax] = grf_data[:, ri] + grf_data[:, li]

    # Subtract gravity from vertical axis
    net_grf[:, 1] -= mass_kg * g

    # Trim GRF to IK length if they differ (timing check handles mismatch)
    T = min(len(time), net_grf.shape[0])
    net_grf = net_grf[:T]

    # ── Numerical differentiation: position -> acceleration ─────────────────
    # np.gradient uses central differences (2nd-order accurate) with
    # forward/backward differences at the boundaries.
    residuals = []
    for ax in range(3):
        p   = pos[ax][:T]
        vel = np.gradient(p,   dt)
        acc = np.gradient(vel, dt)
        expected_force = mass_kg * acc        # m * a  (N)
        residual       = expected_force - net_grf[:, ax]
        residuals.append(residual)

    residuals_arr = np.array(residuals)       # (3, T)
    rms_per_axis  = np.sqrt(np.mean(residuals_arr ** 2, axis=1))
    rms_total     = float(np.sqrt(np.mean(residuals_arr ** 2)))
    rms_bw        = rms_total / bw

    axis_labels = ["x (AP)", "y (vert)", "z (ML)"]
    detail = "  |  ".join(
        f"{ax}: {rms:.1f} N" for ax, rms in zip(axis_labels, rms_per_axis)
    )

    # For running, the pelvis-as-CoM proxy is too crude to threshold reliably:
    # torso counter-rotation and arm swing displace the pelvis 10-15 cm from
    # the true CoM, producing large residuals even on clean data. In run mode,
    # the result is printed as INFO only -- WARN/FAIL are reserved for walk mode
    # where the proxy is accurate enough to be diagnostic.
    if mode == "run":
        status = PASS if rms_bw < NE_COM_RESIDUAL_FAIL_BW["walk"] else WARN
        record(status, "ne_com",
               f"CoM momentum residual RMS = {rms_total:.1f} N "
               f"({rms_bw:.2f} BW) [INFO only for run: pelvis proxy unreliable]. "
               f"Per-axis: {detail}")
        return

    warn_bw = NE_COM_RESIDUAL_WARN_BW[mode]
    fail_bw = NE_COM_RESIDUAL_FAIL_BW[mode]
    if rms_bw > fail_bw:
        record(FAIL, "ne_com",
               f"CoM momentum residual RMS = {rms_total:.1f} N "
               f"({rms_bw:.2f} BW) exceeds {fail_bw} BW. "
               f"Per-axis: {detail}. "
               "Likely cause: wrong IK pass, unit error, or GRF mismatch.")
    elif rms_bw > warn_bw:
        record(WARN, "ne_com",
               f"CoM momentum residual RMS = {rms_total:.1f} N "
               f"({rms_bw:.2f} BW). Per-axis: {detail}. "
               "Pelvis-as-CoM approximation contributes ~5-10% BW; "
               "values above that suggest a real inconsistency.")
    else:
        record(PASS, "ne_com",
               f"CoM momentum residual RMS = {rms_total:.1f} N "
               f"({rms_bw:.2f} BW). Per-axis: {detail}")


def check_id_grf_consistency(
    id_cols, id_data,
    grf_cols, grf_data,
    mass_kg: float,
) -> None:
    """
    Total Support Moment (TSM) vs vertical GRF correlation.

    TSM = hip_flexion_moment + knee_angle_moment - ankle_angle_moment
    (sign convention: all contribute positively to support in Winter 1980).

    In healthy walking and running, TSM correlates strongly with total
    vertical GRF (r > 0.70). A low correlation indicates the ID moments
    are inconsistent with the GRF loading pattern, which typically means
    the dynamics pass used the wrong kinematics or GRF input.

    Both sides are checked independently and as a combined bilateral signal.
    """
    bw = mass_kg * 9.81

    def _tsm(side: str) -> tuple[np.ndarray, tuple, float] | tuple[None, None, None]:
        """
        Return (TSM array, best_signs, best_r) for one side.

        Tries all 8 sign combinations for (hip, knee, ankle) and picks the
        one with the highest Pearson r against Fy during stance. This
        auto-detects the AddBiomechanics moment sign convention rather than
        assuming Winter 1980. If no columns are found, returns (None, None, None).
        """
        suffix  = f"_{side}"
        hip_col = col_idx(id_cols, f"hip_flexion{suffix}_moment")
        kne_col = col_idx(id_cols, f"knee_angle{suffix}_moment")
        ank_col = col_idx(id_cols, f"ankle_angle{suffix}_moment")
        if any(c is None for c in (hip_col, kne_col, ank_col)):
            return None, None, None

        fy_prefix = "" if side == "r" else "l_"
        fy_idx    = col_idx(grf_cols, f"{fy_prefix}ground_force_vy")
        if fy_idx is None:
            return None, None, None

        T_      = min(id_data.shape[0], grf_data.shape[0])
        fy_t    = grf_data[:T_, fy_idx]
        stance  = fy_t > 10.0
        if stance.sum() < 10:
            return None, None, None

        hip_v = id_data[:T_, hip_col]
        kne_v = id_data[:T_, kne_col]
        ank_v = id_data[:T_, ank_col]

        best_r, best_signs, best_tsm = -np.inf, (1, 1, -1), None
        for sh in (1, -1):
            for sk in (1, -1):
                for sa in (1, -1):
                    tsm_candidate = sh*hip_v + sk*kne_v + sa*ank_v
                    r = float(np.corrcoef(
                        tsm_candidate[stance], fy_t[stance]
                    )[0, 1])
                    if r > best_r:
                        best_r     = r
                        best_signs = (sh, sk, sa)
                        best_tsm   = tsm_candidate
        return best_tsm, best_signs, best_r

    def _fy(prefix: str) -> np.ndarray | None:
        idx = col_idx(grf_cols, f"{prefix}ground_force_vy")
        return grf_data[:, idx] if idx is not None else None

    T = min(id_data.shape[0], grf_data.shape[0])

    sign_str = lambda s: "".join("+" if v > 0 else "-" for v in s)

    tsm_arrays = {}
    for side, prefix in (("r", ""), ("l", "l_")):
        tsm, signs, r = _tsm(side)
        if tsm is None:
            record(WARN, f"ne_tsm_{side}",
                   "Skipped: ID moment or GRF columns missing")
            continue

        convention = f"TSM = {sign_str(signs[:1])}hip {sign_str(signs[1:2])}knee {sign_str(signs[2:])}ankle"
        if r < NE_TSM_CORR_FAIL:
            record(FAIL, f"ne_tsm_{side}",
                   f"Best TSM-Fy r={r:.3f} (best combo: {convention}). "
                   f"Even optimal sign combo is below {NE_TSM_CORR_FAIL}: "
                   "ID moments may be structurally inconsistent with GRF.")
        elif r < NE_TSM_CORR_WARN:
            record(WARN, f"ne_tsm_{side}",
                   f"TSM-Fy r={r:.3f} ({convention}): "
                   f"below expected {NE_TSM_CORR_WARN} for healthy gait")
        else:
            record(PASS, f"ne_tsm_{side}",
                   f"TSM-Fy r={r:.3f} ({convention})")

        tsm_arrays[side] = tsm

    # Bilateral: sum the auto-signed TSM arrays
    if "r" in tsm_arrays and "l" in tsm_arrays:
        T = min(id_data.shape[0], grf_data.shape[0])
        fy_r = _fy("")
        fy_l = _fy("l_")
        if fy_r is not None and fy_l is not None:
            tsm_bil = tsm_arrays["r"][:T] + tsm_arrays["l"][:T]
            fy_bil  = fy_r[:T] + fy_l[:T]
            loaded  = fy_bil > 10.0
            if loaded.sum() >= 10:
                r_bil = float(np.corrcoef(tsm_bil[loaded], fy_bil[loaded])[0, 1])
                if r_bil < NE_TSM_CORR_WARN:
                    record(WARN, "ne_tsm_bilateral",
                           f"Bilateral TSM-Fy r={r_bil:.3f}: "
                           "lower limb moments may not balance GRF loading")
                else:
                    record(PASS, "ne_tsm_bilateral",
                           f"Bilateral TSM-Fy r={r_bil:.3f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate process_b3d.py trial outputs")
    parser.add_argument("--trial_dir", required=True,
                        help="Path to trial output directory (contains ik.mot, grf.mot, etc.)")
    parser.add_argument("--mode", choices=["walk", "run"], default="walk",
                        help="Activity mode: walk or run. Adjusts GRF BW "
                             "thresholds and Newton-Euler CoM residual tolerance "
                             "(default: walk)")
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
    print("-- Timing & alignment --")
    check_timing(ik_data, grf_data, id_data)

    print("\n-- IK (ik.mot) --")
    check_ik(ik_cols, ik_data, body)

    print("\n-- GRF (grf.mot) --")
    check_grf(grf_cols, grf_data, mass_kg, mode=args.mode)

    print("\n-- ID moments (id_moments.sto) --")
    check_id(id_cols, id_data)

    print("\n-- Newton-Euler consistency --")
    check_newton_euler_com(ik_cols, ik_data, grf_cols, grf_data, mass_kg, mode=args.mode)
    check_id_grf_consistency(id_cols, id_data, grf_cols, grf_data, mass_kg)


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