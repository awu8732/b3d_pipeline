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


def extract_grf(frames: list, pass_idx: int = 2,
                hold_cop_through_swing: bool = True,
                zero_force_threshold: float = 1e-6) -> np.ndarray:
    """
    Build a (n_frames, 18) bilateral GRF array from processing-pass data.

    Uses the *processed* ground contact fields from the specified pass
    (typically pass 2 / DYNAMICS), which are dynamically consistent with
    the kinematics and tau from that same pass.

    Processing-pass ground contact 6-vector layout:
      groundContactForce            -> (6,)
      groundContactCenterOfPressure -> (6,)
      groundContactTorque           -> (6,)

    Left/right mapping:
      getGroundForceBodies() returns ['calcn_r', 'calcn_l'], so:
        6vec[0:3] = body 0 = calcn_r = anatomical RIGHT -> ground_force_*
        6vec[3:6] = body 1 = calcn_l = anatomical LEFT  -> l_ground_force_*

      An earlier version of this file swapped these based on a
      residual-actuator test that dropped pelvis_ty_reserve from ~826 N to
      near zero. That test was confounded by the swing-frame CoP bug fixed
      below: during swing, pass 2 reports CoP=(0,0,0) for the non-contact
      foot, which produces artificial pelvis moments around heel-strike and
      toe-off transitions that happen to partially cancel under the swapped
      mapping. Re-running the residual test with hold_cop_through_swing=True
      shows the anatomical mapping above is correct.

      Cross-checks confirming this mapping:
        1. CoP lateral position: slot0 stance z is more positive than
           slot1 stance z, matching +z = subject's right in OpenSim's
           default frame (with subject facing +x, pelvis_rotation ~ 0).
        2. Hip flexion during single stance: when only slot1 has force,
           hip_flexion_l is extended (leg trailing) and hip_flexion_r
           is flexed (leg swinging) -> slot1 is anatomical left.
        3. getGroundForceBodies() declaration agrees with both.

    Swing-frame CoP handling:
      Pass 2 reports CoP=(0,0,0) for a foot whenever its Fv=0 (swing phase
      and trial-boundary frames). OpenSim's ID and SO ignore CoP when Fv=0,
      so these zeros are dynamically harmless -- but visualizers draw the
      GRF arrow from the world origin, producing the "floating arrow"
      artifact between the feet. Some downstream tools also mis-handle the
      CoP=(0,0,0) sentinel.

      When hold_cop_through_swing=True (default), this function carries
      each foot's CoP forward from its last active frame across subsequent
      swing frames, and backward-fills any swing frames before that foot's
      first activation. Forces and torques are NOT modified; only CoP
      during Fv=0 frames. This is a pure visualization/downstream-tool fix
      that does not alter ID/SO results.

      Set hold_cop_through_swing=False to preserve the raw pass-2 behaviour
      (zero CoP during swing).

    Column layout matches GRF_COLUMNS (18-column OpenSim bilateral format):
      cols  0-2:  right force   (vx vy vz)
      cols  3-5:  right CoP     (px py pz)
      cols  6-8:  left  force   (vx vy vz)
      cols  9-11: left  CoP     (px py pz)
      cols 12-14: right torque  (tx ty tz)
      cols 15-17: left  torque  (tx ty tz)

    Parameters
    ----------
    frames                 : list of Frame objects
    pass_idx               : processing pass to read from (default 2 = DYNAMICS)
    hold_cop_through_swing : if True, forward/backward-fill CoP across
                             Fv=0 frames (default True)
    zero_force_threshold   : |F| below this is considered a zero-force frame
                             for CoP-fill purposes (default 1e-6 N)

    Returns
    -------
    (n_frames, 18) float64 array
    """
    _ANAT_RIGHT = slice(0, 3)   # 6vec[0:3] = calcn_r = anatomical right
    _ANAT_LEFT  = slice(3, 6)   # 6vec[3:6] = calcn_l = anatomical left

    out = np.zeros((len(frames), 18))
    for fi, frame in enumerate(frames):
        pp  = frame.processingPasses[pass_idx]
        frc = np.array(pp.groundContactForce)
        cop = np.array(pp.groundContactCenterOfPressure)
        trq = np.array(pp.groundContactTorque)

        # Anatomical RIGHT (calcn_r, 6vec[0:3]) -> OpenSim "ground_force_*"
        out[fi, 0:3]   = frc[_ANAT_RIGHT]
        out[fi, 3:6]   = cop[_ANAT_RIGHT]
        # Anatomical LEFT  (calcn_l, 6vec[3:6]) -> OpenSim "l_ground_force_*"
        out[fi, 6:9]   = frc[_ANAT_LEFT]
        out[fi, 9:12]  = cop[_ANAT_LEFT]
        # Torques
        out[fi, 12:15] = trq[_ANAT_RIGHT]
        out[fi, 15:18] = trq[_ANAT_LEFT]

    if hold_cop_through_swing:
        _fill_swing_cop(out, (slice(0, 3), slice(3, 6)),   # right: force, CoP
                        zero_force_threshold)
        _fill_swing_cop(out, (slice(6, 9), slice(9, 12)),  # left:  force, CoP
                        zero_force_threshold)

    return out


def _fill_swing_cop(out: np.ndarray, slices: tuple,
                    threshold: float) -> None:
    """
    In-place: carry CoP forward through frames where the foot's force is
    below `threshold`. Backward-fills any leading swing frames before the
    first active frame. If the foot is never active in this trial, CoP is
    left as zeros (there is no sensible value to fill with).

    `slices` is (force_slice, cop_slice). Each slice is a 3-element slice
    into the 18-column GRF array for one foot (force v and CoP p).
    """
    force_slice, cop_slice = slices
    force_mag = np.linalg.norm(out[:, force_slice], axis=1)
    active    = force_mag > threshold
    if not active.any():
        return  # foot never loaded in this trial -- leave zeros

    first = int(np.argmax(active))
    last_cop = out[first, cop_slice].copy()
    for i in range(first, len(out)):
        if active[i]:
            last_cop = out[i, cop_slice].copy()
        else:
            out[i, cop_slice] = last_cop

    # Backward-fill frames before first activation with the first active CoP.
    if first > 0:
        out[:first, cop_slice] = out[first, cop_slice]


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