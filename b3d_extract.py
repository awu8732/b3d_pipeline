"""
b3d_extract.py
==============
Frame-level data extraction from nimblephysics SubjectOnDisk objects.

Pulls out markers, IK positions, ID moments, and GRF arrays from the
flat list of Frame objects returned by SubjectOnDisk.readFrames().
Previously inlined inside process_b3d.py.

GRF extraction uses the *processed* ground contact fields from the
specified processing pass (default: pass 2 / DYNAMICS), NOT the raw
force plate readings. The raw readings (rawForcePlateForces, etc.)
may differ from the processed values and are not dynamically consistent
with the pass kinematics/tau. The old raw extractor is preserved as
extract_grf_raw() for comparison only.

Ground contact body order in the 6-vector fields:
  index 0:3 = calcn_l (left)   -- first body packed
  index 3:6 = calcn_r (right)  -- second body packed
"""

from __future__ import annotations

import numpy as np

# Standard OpenSim bilateral 18-column GRF layout:
#   right: force (vx vy vz) + CoP (px py pz) + free torque (tx ty tz)
#   left:  force (vx vy vz) + CoP (px py pz) + free torque (tx ty tz)
GRF_COLUMNS = [
    "ground_force_vx",   "ground_force_vy",   "ground_force_vz",
    "ground_force_px",   "ground_force_py",   "ground_force_pz",
    "l_ground_force_vx", "l_ground_force_vy", "l_ground_force_vz",
    "l_ground_force_px", "l_ground_force_py", "l_ground_force_pz",
    "ground_torque_x",   "ground_torque_y",   "ground_torque_z",
    "l_ground_torque_x", "l_ground_torque_y", "l_ground_torque_z",
]

_GRF_RIGHT = 0
_GRF_LEFT  = 1


def read_all_frames(subject, trial_idx: int, n_frames: int,
                    batch_size: int = 512) -> list:
    """
    Read all frames for a trial in batches to limit peak RAM usage.

    Parameters
    ----------
    subject    : nimblephysics SubjectOnDisk instance
    trial_idx  : zero-based trial index
    n_frames   : total frame count for this trial
    batch_size : frames per readFrames() call

    Returns
    -------
    Flat list of Frame objects (length == n_frames)
    """
    frames = []
    start  = 0
    while start < n_frames:
        batch_n = min(batch_size, n_frames - start)
        batch   = subject.readFrames(
            trial                   = trial_idx,
            startFrame              = start,
            numFramesToRead         = batch_n,
            includeSensorData       = True,
            includeProcessingPasses = True,
        )
        frames.extend(batch)
        start += batch_n
    return frames


def extract_markers(frames: list, marker_names: list,
                    verbose: bool = True) -> np.ndarray:
    """
    Build a (n_frames, n_markers * 3) marker position array.

    Missing observations are filled with NaN and then linearly interpolated
    column-wise so downstream filters never see gaps. Leading/trailing NaN
    runs are extended with the nearest valid sample (np.interp's default
    behaviour is to clamp, which is what we want; this is made explicit via
    `left` / `right` args for clarity). Fully-empty columns (marker never
    observed) are zero-filled and reported; otherwise `sosfiltfilt` would
    propagate NaN through the entire trial.

    Parameters
    ----------
    frames       : list of Frame objects
    marker_names : ordered list of expected marker labels
    verbose      : print per-column gap / missing-marker warnings

    Returns
    -------
    out : (n_frames, n_markers * 3) float64 array, XYZ interleaved
    """
    n = len(frames)
    m = len(marker_names)
    name_to_idx = {name: i for i, name in enumerate(marker_names)}
    out = np.full((n, m * 3), np.nan)

    for fi, frame in enumerate(frames):
        for name, xyz in frame.markerObservations:
            if name in name_to_idx:
                idx = name_to_idx[name]
                out[fi, idx * 3 : idx * 3 + 3] = xyz

    # Per-marker gap bookkeeping for a single per-marker warning line
    n_missing_markers = 0
    max_gap_frames    = 0
    gap_frame_total   = 0

    for col in range(out.shape[1]):
        mask = np.isnan(out[:, col])
        if not mask.any():
            continue
        valid = np.where(~mask)[0]
        if len(valid) == 0:
            # Marker never observed in this trial: zero-fill so filtering works.
            out[:, col] = 0.0
            if col % 3 == 0 and verbose:
                n_missing_markers += 1
                print(f"    [markers] '{marker_names[col // 3]}' never observed"
                      f" -> zero-filled (sensor-wise; will not move)")
            continue
        # Interpolate interior + clamp edges to first/last valid sample.
        out[:, col] = np.interp(
            np.arange(n), valid, out[valid, col],
            left=out[valid[0], col], right=out[valid[-1], col],
        )
        # Gap stats (X channel only, to count once per marker-frame).
        if col % 3 == 0:
            gap_frame_total += int(mask.sum())
            # Longest contiguous run of NaNs in this column
            if mask.any():
                pad    = np.r_[False, mask, False]
                deltas = np.diff(pad.astype(int))
                starts = np.where(deltas ==  1)[0]
                ends   = np.where(deltas == -1)[0]
                if len(starts):
                    max_gap_frames = max(max_gap_frames,
                                         int((ends - starts).max()))

    if verbose and (gap_frame_total or n_missing_markers):
        print(f"    [markers] interpolated {gap_frame_total} missing frame-"
              f"observations (longest contiguous gap: {max_gap_frames} frames)")

    return out


def extract_ik(frames: list, n_dofs: int, pass_idx: int) -> np.ndarray:
    """
    Extract joint positions (radians) from the specified processing pass.

    Returns
    -------
    (n_frames, n_dofs) float64 array
    """
    out = np.zeros((len(frames), n_dofs))
    for fi, frame in enumerate(frames):
        out[fi] = frame.processingPasses[pass_idx].pos
    return out


def extract_id(frames: list, n_dofs: int, pass_idx: int) -> np.ndarray:
    """
    Extract joint moments (N·m) from the specified processing pass.

    Returns
    -------
    (n_frames, n_dofs) float64 array
    """
    out = np.zeros((len(frames), n_dofs))
    for fi, frame in enumerate(frames):
        out[fi] = frame.processingPasses[pass_idx].tau
    return out


def extract_grf(frames: list, pass_idx: int = 2) -> np.ndarray:
    """
    Build a (n_frames, 18) bilateral GRF array from processing-pass data.

    Uses the *processed* ground contact fields from the specified pass
    (typically pass 2 / DYNAMICS), which are dynamically consistent with
    the kinematics and tau from that same pass.

    Processing-pass ground contact 6-vector layout:
      groundContactForce            -> (6,)
      groundContactCenterOfPressure -> (6,)
      groundContactTorque           -> (6,)

    Left/right mapping (empirically confirmed):
      getGroundForceBodies() returns ['calcn_r', 'calcn_l'], suggesting
      body 0 = calcn_r and body 1 = calcn_l. However, cross-referencing
      single-support frames against IK joint angles reveals that when
      contact=[0,1] (body 1 active), the stance leg is anatomically RIGHT
      (hip extending, knee flexed on the right side). This means the
      nimblephysics body labels are swapped relative to the OpenSim
      skeleton's anatomical convention.

      Mapping confirmed by reserve actuator magnitudes: with the correct
      mapping, pelvis_ty_reserve drops from ~826 N (107% BW) to near zero,
      and ankle/hip reserves become physiologically plausible.

      6vec[0:3] = body 0 = anatomical LEFT  -> OpenSim l_ground_force_*
      6vec[3:6] = body 1 = anatomical RIGHT -> OpenSim ground_force_*

    Column layout matches GRF_COLUMNS (18-column OpenSim bilateral format):
      cols  0-2:  right force   (vx vy vz)
      cols  3-5:  right CoP     (px py pz)
      cols  6-8:  left  force   (vx vy vz)
      cols  9-11: left  CoP     (px py pz)
      cols 12-14: right torque  (tx ty tz)
      cols 15-17: left  torque  (tx ty tz)

    Parameters
    ----------
    frames   : list of Frame objects
    pass_idx : processing pass to read from (default 2 = DYNAMICS)

    Returns
    -------
    (n_frames, 18) float64 array
    """
    _ANAT_LEFT  = slice(0, 3)   # 6vec[0:3] = anatomical left foot
    _ANAT_RIGHT = slice(3, 6)   # 6vec[3:6] = anatomical right foot

    out = np.zeros((len(frames), 18))
    for fi, frame in enumerate(frames):
        pp  = frame.processingPasses[pass_idx]
        frc = np.array(pp.groundContactForce)
        cop = np.array(pp.groundContactCenterOfPressure)
        trq = np.array(pp.groundContactTorque)

        # Anatomical RIGHT -> OpenSim "ground_force_*" columns
        out[fi, 0:3]   = frc[_ANAT_RIGHT]
        out[fi, 3:6]   = cop[_ANAT_RIGHT]
        # Anatomical LEFT  -> OpenSim "l_ground_force_*" columns
        out[fi, 6:9]   = frc[_ANAT_LEFT]
        out[fi, 9:12]  = cop[_ANAT_LEFT]
        # Torques
        out[fi, 12:15] = trq[_ANAT_RIGHT]
        out[fi, 15:18] = trq[_ANAT_LEFT]
    return out


def extract_grf_raw(frames: list) -> np.ndarray:
    """
    [DEPRECATED] Build GRF array from raw (unprocessed) force plate data.

    Retained for comparison/debugging. For SO-compatible output, use
    extract_grf() which reads from the dynamics processing pass instead.

    Returns
    -------
    (n_frames, 18) float64 array
    """
    out = np.zeros((len(frames), 18))
    for fi, frame in enumerate(frames):
        rf = frame.rawForcePlateForces
        rc = frame.rawForcePlateCenterOfPressures
        rt = frame.rawForcePlateTorques
        out[fi, 0:3]   = rf[_GRF_RIGHT]
        out[fi, 3:6]   = rc[_GRF_RIGHT]
        out[fi, 6:9]   = rf[_GRF_LEFT]
        out[fi, 9:12]  = rc[_GRF_LEFT]
        out[fi, 12:15] = rt[_GRF_RIGHT]
        out[fi, 15:18] = rt[_GRF_LEFT]
    return out