"""
b3d_grf_qc.py
=============
Quality-control and correction for GRF center-of-pressure data.

Two operations:
  1. flag_cop_outliers()    -- always-on diagnostic. Per-frame distance from
                               CoP to a foot polygon defined relative to the
                               calcaneus body. Returns flags + summary stats
                               with no side effects on the GRF array.
  2. correct_cop_outliers() -- opt-in correction. For frames whose CoP is
                               more than `threshold_m` outside the foot
                               polygon, project the CoP onto the nearest
                               polygon edge. Force vector and free torque
                               are NOT modified. Returns a new GRF array.

Foot polygon model
------------------
OpenSim biomechanical models don't carry explicit foot-shape geometry that
can be queried per frame, so we use a per-side rectangular polygon defined
in the calcaneus body frame:

  AP (calcn x): heel at FOOT_AP_HEEL, toe at FOOT_AP_TOE
  ML (calcn z): medial / lateral edges at FOOT_ML_MEDIAL / FOOT_ML_LATERAL
                with the sign flipping per side (calcn_l vs calcn_r)

Defaults are based on a 70-kg male reference foot (~26 cm long, ~10 cm wide
at the metatarsals) and are scaled per-trial by the model's body_scales for
calcn_<side>. If body_scales for the foot are unavailable (1.0 default),
the unscaled polygon is used.

Output to .mot is unaffected by flagging; correction (when enabled) writes
a modified CoP into the GRF array before write_mot() is called.

Sidecar JSON layout
-------------------
A grf_qc.json is written next to grf.mot containing:
  threshold_m              float
  correction_applied       bool
  foot_polygon_calcn_r     {ap_heel, ap_toe, ml_medial, ml_lateral}  (metres)
  foot_polygon_calcn_l     same
  per_frame_distance_r     [T] metres (0 if inside polygon, +ve if outside)
  per_frame_distance_l     [T]
  flagged_frame_count_r    int  (frames with d > threshold_m on right)
  flagged_frame_count_l    int
  flagged_frame_count_total int (union; frame counted once if either bad)
  total_stance_frames_r    int  (frames with right Fv > stance_force_n)
  total_stance_frames_l    int
  flagged_fraction_r       float  (= flagged_r / max(stance_r, 1))
  flagged_fraction_l       float
  trial_max_distance_r     float  metres
  trial_max_distance_l     float
  trial_mean_distance_r    float  (mean over stance frames only)
  trial_mean_distance_l    float

This lets downstream training-data assembly drop or weight trials by
the QC stats without re-running the analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── Foot polygon in calcaneus body frame ─────────────────────────────────────
# Positive x = anterior, positive z = anatomically lateral (subject's right
# direction in world for calcn_r when subject faces +x). The ML signs flip
# per side because the calcaneus body frame's z-axis points laterally for
# both sides in the standard Rajagopal/LaiArnold convention.

FOOT_AP_HEEL    = -0.05   # 5 cm posterior of calcn origin
FOOT_AP_TOE     =  0.22   # 22 cm anterior (covers metatarsal heads + toes)
FOOT_ML_HALF    =  0.05   # 5 cm to either side of calcn x-axis

# Below this Fv (N), a foot is considered in swing and CoP is not evaluated.
DEFAULT_STANCE_FORCE_N = 20.0


def _foot_polygon(side: str, scale_ap: float = 1.0,
                  scale_ml: float = 1.0) -> dict:
    """
    Return the per-side foot polygon (4 edges as min/max bounds in calcn frame).

    The polygon is rectangular in the AP-ML plane. ML half-width applies
    symmetrically about the calcn x-axis.
    """
    return {
        "ap_heel":    FOOT_AP_HEEL * scale_ap,
        "ap_toe":     FOOT_AP_TOE  * scale_ap,
        "ml_medial":  -FOOT_ML_HALF * scale_ml,  # negative side
        "ml_lateral":  FOOT_ML_HALF * scale_ml,  # positive side
    }


def _point_to_rect_distance_and_proj(p_ap: float, p_ml: float,
                                     poly: dict) -> tuple[float, float, float]:
    """
    Distance from a point (p_ap, p_ml) in calcn frame to the rectangle
    defined by `poly`, plus the projection (closest point on the polygon
    or inside it) as (proj_ap, proj_ml).

    If the point is inside the rectangle, distance is 0 and projection
    equals the point itself.
    """
    proj_ap = min(max(p_ap, poly["ap_heel"]),    poly["ap_toe"])
    proj_ml = min(max(p_ml, poly["ml_medial"]),  poly["ml_lateral"])
    d = float(np.hypot(p_ap - proj_ap, p_ml - proj_ml))
    return d, proj_ap, proj_ml


def _world_to_calcn_per_frame(skel, ik_rad: np.ndarray,
                              body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward kinematics frame-by-frame to get the calcaneus body's
    world-frame origin and rotation matrix.

    Parameters
    ----------
    skel     : nimble Skeleton object
    ik_rad   : (T, n_dofs) joint positions in radians, in skeleton DOF order
    body_name: e.g. 'calcn_r' or 'calcn_l'

    Returns
    -------
    origin_w : (T, 3) calcn body origin in world coordinates
    R_wb     : (T, 3, 3) rotation from body frame to world frame
    """
    T = ik_rad.shape[0]
    origin_w = np.zeros((T, 3))
    R_wb     = np.zeros((T, 3, 3))

    body = skel.getBodyNode(body_name)
    for fi in range(T):
        skel.setPositions(ik_rad[fi])
        Tw = body.getWorldTransform()
        # nimble uses Eigen-style API; .translation() and .rotation() return
        # numpy arrays in modern nimblephysics builds.
        origin_w[fi] = np.asarray(Tw.translation()).flatten()
        R_wb[fi]     = np.asarray(Tw.rotation())
    return origin_w, R_wb


def _cop_in_foot_frame(cop_world: np.ndarray, origin_w: np.ndarray,
                       R_wb: np.ndarray) -> np.ndarray:
    """
    Express world-frame CoP in the body frame given per-frame body origin
    and rotation.

    cop_body = R_wb^T @ (cop_world - origin_w)
    """
    rel = cop_world - origin_w                          # (T, 3)
    # einsum: for each frame, R_wb^T @ rel -> 3-vector
    return np.einsum("tji,tj->ti", R_wb, rel)


# ── Public API ───────────────────────────────────────────────────────────────

def flag_cop_outliers(
    grf:        np.ndarray,
    skel,
    ik_rad:     np.ndarray,
    body_scales: dict | None = None,
    threshold_m: float       = 0.03,
    stance_force_n: float    = DEFAULT_STANCE_FORCE_N,
) -> dict:
    """
    Diagnose CoP-foot alignment without modifying GRF.

    Parameters
    ----------
    grf           : (T, 18) GRF array from extract_grf() (post hold-CoP fix)
    skel          : nimble Skeleton in current model state (will be mutated;
                    caller should restore positions afterwards if needed)
    ik_rad        : (T, n_dofs) joint positions in radians (skeleton DOF order)
    body_scales   : optional dict body_name -> (sx, sy, sz) per-axis scales.
                    If None, no scaling is applied (uses unscaled defaults).
    threshold_m   : flag distance threshold (default 3 cm)
    stance_force_n: vertical force threshold below which a frame is not
                    evaluated for that side (swing phase / boundary)

    Returns
    -------
    dict matching the sidecar JSON layout. See module docstring.
    """
    T = grf.shape[0]

    # Per-side AP / ML scales from body_scales if available. body_scales
    # commonly comes as a 3-element vector for each body in OpenSim convention
    # (sx, sy, sz). Calcaneus AP is x, ML is z.
    def _scales(body_name: str) -> tuple[float, float]:
        if body_scales is None or body_name not in body_scales:
            return 1.0, 1.0
        v = body_scales[body_name]
        try:
            return float(v[0]), float(v[2])
        except (TypeError, IndexError):
            return 1.0, 1.0

    sap_r, sml_r = _scales("calcn_r")
    sap_l, sml_l = _scales("calcn_l")
    poly_r = _foot_polygon("r", sap_r, sml_r)
    poly_l = _foot_polygon("l", sap_l, sml_l)

    # Forces per side (used to identify stance vs swing).
    fy_r = grf[:, 1]    # ground_force_vy
    fy_l = grf[:, 7]    # l_ground_force_vy

    # CoP world-frame columns.
    cop_r_w = grf[:, 3:6].copy()   # ground_force_p[xyz]
    cop_l_w = grf[:, 9:12].copy()  # l_ground_force_p[xyz]

    # FK to get foot poses (right and left).
    origin_r_w, R_wb_r = _world_to_calcn_per_frame(skel, ik_rad, "calcn_r")
    origin_l_w, R_wb_l = _world_to_calcn_per_frame(skel, ik_rad, "calcn_l")

    cop_r_f = _cop_in_foot_frame(cop_r_w, origin_r_w, R_wb_r)
    cop_l_f = _cop_in_foot_frame(cop_l_w, origin_l_w, R_wb_l)

    # Per-frame distance: 0 inside polygon, positive outside, NaN in swing.
    dist_r = np.full(T, np.nan)
    dist_l = np.full(T, np.nan)
    stance_r = fy_r > stance_force_n
    stance_l = fy_l > stance_force_n

    for i in np.where(stance_r)[0]:
        d, _, _ = _point_to_rect_distance_and_proj(
            cop_r_f[i, 0], cop_r_f[i, 2], poly_r)
        dist_r[i] = d
    for i in np.where(stance_l)[0]:
        d, _, _ = _point_to_rect_distance_and_proj(
            cop_l_f[i, 0], cop_l_f[i, 2], poly_l)
        dist_l[i] = d

    flagged_r = np.where(dist_r > threshold_m)[0]
    flagged_l = np.where(dist_l > threshold_m)[0]
    flagged_total = np.union1d(flagged_r, flagged_l)

    def _safe_max(a):
        a = a[~np.isnan(a)]
        return float(a.max()) if a.size else 0.0
    def _safe_mean(a):
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else 0.0
    def _safe_median(a):
        a = a[~np.isnan(a)]
        return float(np.median(a)) if a.size else 0.0

    return {
        "threshold_m":               threshold_m,
        "correction_applied":        False,
        "foot_polygon_calcn_r":      poly_r,
        "foot_polygon_calcn_l":      poly_l,
        "per_frame_distance_r":      [None if np.isnan(x) else float(x) for x in dist_r],
        "per_frame_distance_l":      [None if np.isnan(x) else float(x) for x in dist_l],
        "flagged_frame_count_r":     int(len(flagged_r)),
        "flagged_frame_count_l":     int(len(flagged_l)),
        "flagged_frame_count_total": int(len(flagged_total)),
        "total_stance_frames_r":     int(stance_r.sum()),
        "total_stance_frames_l":     int(stance_l.sum()),
        "flagged_fraction_r":        float(len(flagged_r) / max(stance_r.sum(), 1)),
        "flagged_fraction_l":        float(len(flagged_l) / max(stance_l.sum(), 1)),
        "trial_max_distance_r":      _safe_max(dist_r),
        "trial_max_distance_l":      _safe_max(dist_l),
        "trial_mean_distance_r":     _safe_mean(dist_r),
        "trial_mean_distance_l":     _safe_mean(dist_l),
        "trial_median_distance_r":   _safe_median(dist_r),
        "trial_median_distance_l":   _safe_median(dist_l),
        # Cached internals used by correct_cop_outliers (not serialized).
        "_internals": {
            "cop_r_f":    cop_r_f,
            "cop_l_f":    cop_l_f,
            "origin_r_w": origin_r_w,
            "origin_l_w": origin_l_w,
            "R_wb_r":     R_wb_r,
            "R_wb_l":     R_wb_l,
            "stance_r":   stance_r,
            "stance_l":   stance_l,
            "poly_r":     poly_r,
            "poly_l":     poly_l,
        },
    }


def correct_cop_outliers(
    grf:        np.ndarray,
    qc:         dict,
    threshold_m: float = 0.03,
) -> tuple[np.ndarray, int]:
    """
    Project CoP onto the foot polygon for frames where distance > threshold.

    Only frames flagged as outliers are modified; in-polygon frames and
    frames within threshold are left untouched. Force vector and free
    torque columns are not modified.

    Vertical CoP (calcn-y in body frame, which becomes pelvis-relative when
    transformed back) is preserved -- only AP and ML are projected.

    Parameters
    ----------
    grf         : (T, 18) GRF array
    qc          : output of flag_cop_outliers() (must include _internals)
    threshold_m : projection threshold (default 3 cm)

    Returns
    -------
    grf_corrected : (T, 18) array with CoP columns corrected where flagged
    n_corrected   : total number of (frame, side) corrections applied
    """
    out = grf.copy()
    intl = qc["_internals"]
    n_corrected = 0

    for side, cop_f, origin_w, R_wb, stance, poly, cop_cols in [
        ("r", intl["cop_r_f"], intl["origin_r_w"], intl["R_wb_r"],
              intl["stance_r"], intl["poly_r"], slice(3, 6)),
        ("l", intl["cop_l_f"], intl["origin_l_w"], intl["R_wb_l"],
              intl["stance_l"], intl["poly_l"], slice(9, 12)),
    ]:
        for i in np.where(stance)[0]:
            d, proj_ap, proj_ml = _point_to_rect_distance_and_proj(
                cop_f[i, 0], cop_f[i, 2], poly)
            if d <= threshold_m:
                continue
            # Project in body frame, keep body-frame y unchanged.
            cop_body_corrected = np.array(
                [proj_ap, cop_f[i, 1], proj_ml]
            )
            # Transform back to world: world = origin + R_wb @ body
            cop_world_new = origin_w[i] + R_wb[i] @ cop_body_corrected
            out[i, cop_cols] = cop_world_new
            n_corrected += 1

    return out, n_corrected


def write_qc_sidecar(path: Path, qc: dict) -> None:
    """Write QC results to JSON, stripping internals."""
    serializable = {k: v for k, v in qc.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"    [json] grf qc -> {path.name}")