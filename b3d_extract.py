"""
b3d_extract.py
==============
Frame-level data extraction from nimblephysics SubjectOnDisk objects.

Pulls out markers, IK positions, ID moments, and GRF arrays from the
flat list of Frame objects returned by SubjectOnDisk.readFrames().
Previously inlined inside process_b3d.py.

GRF index convention (confirmed against getGroundForceBodies()):
  index 0 = calcn_r (right)
  index 1 = calcn_l (left)
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


def extract_markers(frames: list, marker_names: list) -> np.ndarray:
    """
    Build a (n_frames, n_markers * 3) marker position array.

    Missing observations are filled with NaN and then linearly interpolated
    column-wise so downstream filters never see gaps.

    Parameters
    ----------
    frames       : list of Frame objects
    marker_names : ordered list of expected marker labels

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

    # Linear interpolation for any NaN columns (missing frames)
    for col in range(out.shape[1]):
        mask = np.isnan(out[:, col])
        if mask.any():
            valid = np.where(~mask)[0]
            if len(valid):
                out[:, col] = np.interp(np.arange(n), valid, out[valid, col])

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


def extract_grf(frames: list) -> np.ndarray:
    """
    Build a (n_frames, 18) bilateral GRF array from raw force plate data.

    Source fields per frame:
      rawForcePlateForces[i]            -> (3,)  [Fx Fy Fz]
      rawForcePlateCenterOfPressures[i] -> (3,)  [px py pz]
      rawForcePlateTorques[i]           -> (3,)  [tx ty tz]

    Column layout matches GRF_COLUMNS (18-column OpenSim bilateral format).

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
