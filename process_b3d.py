"""
process_b3d.py
==============
Single-subject B3D processing pipeline using nimblephysics.

Confirmed API against nimblephysics (Hammer2013 dataset, 3-pass B3D):
  Pass 0: KINEMATICS       -- raw IK positions
  Pass 1: LOW_PASS_FILTER  -- AddBiomechanics-filtered positions (used as IK source)
  Pass 2: DYNAMICS         -- joint moments (tau), used for ID output

Outputs per trial (OUTPUT_ROOT / <subject_tag> / <trial_name> /):
  markers.trc      -- low-pass filtered 3D marker trajectories
  ik.mot           -- IK joint angles (pass 1 pos + optional extra LPF), degrees
  id_moments.sto   -- ID joint moments from pass 2 tau, N·m
  grf.mot          -- bilateral GRF + CoP, single OpenSim 18-col layout
  body.json        -- subject-level body parameters

Dependencies:
  pip install nimblephysics scipy numpy

Internal modules (same directory):
  b3d_io.py       -- file writers/readers
  b3d_filters.py  -- Butterworth LPF helpers
  b3d_extract.py  -- frame-level data extraction
"""

import os
import sys
from pathlib import Path

import nimblephysics as nimble
import numpy as np

from b3d_io      import write_trc, write_mot, write_sto, write_body_json
from b3d_filters import make_lpf, apply_lpf
from b3d_extract import (
    read_all_frames, extract_markers, extract_ik, extract_id, extract_grf,
    GRF_COLUMNS,
)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG: edit this block; leave everything else alone
# ════════════════════════════════════════════════════════════════════════════

B3D_PATH    = "AddBiomechanicsDataset/test/With_Arm/Hammer2013_Formatted_With_Arm/subject10/subject10.b3d"
OUTPUT_ROOT = "output"

# Sampling rate (Hz). The actual per-trial timestep is read from the file;
# SAMPLE_RATE_HZ is used only for filter design and as a fallback if
# getTrialTimestep() returns 0.
SAMPLE_RATE_HZ = 100

# Processing pass indices (0-based).
# Confirmed order for AddBiomechanics 3-pass B3D: 0=KIN, 1=LPF, 2=DYN.
IK_PASS_IDX = 1   # LOW_PASS_FILTER pass: source for joint positions
ID_PASS_IDX = 2   # DYNAMICS pass: source for joint moments (tau)

# Low-pass filter applied to markers after extraction (Stage 1).
LPF_MARKERS_HZ    = 6.0   # cutoff frequency, Hz
LPF_MARKERS_ORDER = 4     # Butterworth order (zero-phase doubles effective order)

# Low-pass filter applied to IK positions from pass 1 (Stage 2).
# Pass 1 is already filtered by AddBiomechanics at their stored cutoff.
# Set LPF_IK_HZ to None to skip this second filter and use pass 1 as-is.
LPF_IK_HZ    = None   # e.g. 6.0 to re-filter, or None to skip
LPF_IK_ORDER = 4

# Frames to read per batch (reduce if RAM is tight).
READ_BATCH_SIZE = 512

# ════════════════════════════════════════════════════════════════════════════


# ── IK post-processing ────────────────────────────────────────────────────────

def _append_beta_coords(ik_out: np.ndarray, dof_names: list) -> tuple[np.ndarray, list]:
    """
    Append patellofemoral beta coordinates required by the LaiArnold model.

    The CoordinateCouplerConstraint uses a LinearFunction with coefficients
    [1, 0], so beta equals the corresponding knee angle at every frame.
    Returns the augmented array and the extended dof name list.
    """
    beta_cols = {
        "knee_angle_r_beta": "knee_angle_r",
        "knee_angle_l_beta": "knee_angle_l",
    }
    beta_data = {}
    for beta_name, source_name in beta_cols.items():
        if beta_name not in dof_names and source_name in dof_names:
            src_idx = dof_names.index(source_name)
            beta_data[beta_name] = ik_out[:, src_idx].copy()

    if not beta_data:
        return ik_out, dof_names

    extra = np.column_stack([np.deg2rad(vals) for vals in beta_data.values()])
    ik_out = np.hstack([ik_out, extra])
    print(f"    Appended beta coords (radians): {list(beta_data.keys())}")
    return ik_out, dof_names + list(beta_data.keys())


def _rad_to_deg_ik(ik_rad: np.ndarray, dof_names: list) -> np.ndarray:
    """
    Convert rotational DOFs from radians to degrees; leave pelvis translations
    in metres.
    """
    tx_cols  = [i for i, name in enumerate(dof_names)
                if name in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    rot_cols = [i for i in range(ik_rad.shape[1]) if i not in tx_cols]

    ik_out = ik_rad.copy()
    ik_out[:, rot_cols] = np.degrees(ik_rad[:, rot_cols])
    if tx_cols:
        print(
            f"    Pelvis translations kept in metres "
            f"(cols {tx_cols}: {[dof_names[i] for i in tx_cols]})"
        )
    return ik_out


# ── Subject-level setup ───────────────────────────────────────────────────────

def _load_subject_metadata(subject, b3d_path: str) -> dict:
    """
    Extract subject-level metadata (name, skeleton, marker names, body params).
    Returns a dict with keys: subject_name, dof_names, moment_names,
    marker_names, body_params, skel.
    """
    subject_name = Path(b3d_path).parent.name
    n_dofs       = subject.getNumDofs()
    href         = subject.getHref()
    mass_kg      = subject.getMassKg()
    height_m     = subject.getHeightM()

    print(f"Subject : {subject_name}")
    print(f"Href    : {href}")
    print(f"Trials  : {subject.getNumTrials()}    DOFs: {n_dofs}")
    print(f"Body    : {mass_kg:.1f} kg, {height_m:.3f} m")

    # readSkel() emits C++ geometry-loading warnings to stderr; suppress them.
    with open(os.devnull, "w") as _devnull:
        _old, sys.stderr = sys.stderr, _devnull
        try:
            skel = subject.readSkel(0)
        finally:
            sys.stderr = _old

    dof_names    = [dof.getName() for dof in skel.getDofs()]
    moment_names = [f"{d}_moment" for d in dof_names]

    # Marker names from the first frame of the first trial
    probe = subject.readFrames(
        trial=0, startFrame=0, numFramesToRead=1,
        includeSensorData=True, includeProcessingPasses=False,
    )
    marker_names = [name for name, _ in probe[0].markerObservations]
    print(f"Markers : {len(marker_names)}")
    print(f"DOFs    : {dof_names}")

    body_params = {
        "subject_name": subject_name,
        "href":         href,
        "mass_kg":      mass_kg,
        "height_m":     height_m,
        "dof_names":    dof_names,
        "marker_names": marker_names,
        "grf_bodies":   list(subject.getGroundForceBodies()),
    }
    try:
        body_params["body_scales"] = skel.getBodyScales().tolist()
    except Exception:
        body_params["body_scales"] = {}

    return dict(
        subject_name = subject_name,
        n_dofs       = n_dofs,
        dof_names    = dof_names,
        moment_names = moment_names,
        marker_names = marker_names,
        body_params  = body_params,
        skel         = skel,
    )


# ── Per-trial pipeline ────────────────────────────────────────────────────────

def process_trial(
    subject,
    trial_idx: int,
    out_dir: Path,
    meta: dict,
    sos_markers,
) -> None:
    """
    Run the full extraction + write pipeline for a single trial.

    Stages:
      1. Markers   -> markers.trc
      2. IK        -> ik.mot
      3a. ID       -> id_moments.sto
      3b. GRF      -> grf.mot
      3c. Body     -> body.json
    """
    trial_name = subject.getTrialName(trial_idx) or f"trial_{trial_idx:02d}"
    n_frames   = subject.getTrialLength(trial_idx)
    timestep   = subject.getTrialTimestep(trial_idx)
    fs         = (1.0 / timestep) if timestep > 0 else float(SAMPLE_RATE_HZ)

    print(f"\n{'-'*50}")
    print(f"  Trial {trial_idx}: {trial_name}")
    print(f"  Frames: {n_frames}  fs: {fs:.4f} Hz  ({n_frames / fs:.2f} s)")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Reading frames...", end=" ", flush=True)
    frames = read_all_frames(subject, trial_idx, n_frames, READ_BATCH_SIZE)
    print(f"{len(frames)} loaded.")

    n_dofs       = meta["n_dofs"]
    dof_names    = meta["dof_names"]
    moment_names = meta["moment_names"]
    marker_names = meta["marker_names"]

    # Stage 1: markers
    print("  [Stage 1] Markers...")
    raw_markers      = extract_markers(frames, marker_names)
    filtered_markers = apply_lpf(raw_markers, sos_markers)
    write_trc(out_dir / "markers.trc", marker_names, filtered_markers, fs)

    # Stage 2: IK
    print("  [Stage 2] IK (pass 1: LOW_PASS_FILTER)...")
    ik_rad = extract_ik(frames, n_dofs, IK_PASS_IDX)
    if LPF_IK_HZ is not None:
        sos_ik = make_lpf(LPF_IK_HZ, LPF_IK_ORDER, fs)
        ik_rad = apply_lpf(ik_rad, sos_ik)
        print(f"    Additional LPF applied: {LPF_IK_HZ} Hz order {LPF_IK_ORDER}")
    else:
        print("    No additional LPF (pass 1 used as-is).")

    ik_out                 = _rad_to_deg_ik(ik_rad, dof_names)
    ik_out, aug_dof_names  = _append_beta_coords(ik_out, dof_names)

    write_mot(
        out_dir / "ik.mot", aug_dof_names,
        ik_out, fs,
        header_name="Coordinates", in_degrees=True,
    )

    # Stage 3a: ID moments
    print("  [Stage 3a] ID moments (pass 2: DYNAMICS)...")
    tau = extract_id(frames, n_dofs, ID_PASS_IDX)
    write_sto(
        out_dir / "id_moments.sto", moment_names, tau, fs,
        header_name="InverseDynamics",
    )

    # Stage 3b: GRF
    print("  [Stage 3b] GRF...")
    grf = extract_grf(frames)
    write_mot(
        out_dir / "grf.mot", GRF_COLUMNS, grf, fs,
        header_name="GRF", in_degrees=False,
    )

    # Stage 3c: body params
    write_body_json(out_dir / "body.json", meta["body_params"])


# ── Top-level entry point ─────────────────────────────────────────────────────

def process_subject(b3d_path: str, output_root: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Loading : {b3d_path}")
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    meta = _load_subject_metadata(subject, b3d_path)

    subject_out_dir = output_root / meta["subject_name"]
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    # Write shared .osim model once at the subject level
    osim_text = subject.getOpensimFileText(0)
    osim_path = subject_out_dir / f"{meta['subject_name']}.osim"
    osim_path.write_text(osim_text)
    print(f"  [osim] model -> {osim_path}")

    # Design the marker filter once (fs fallback; per-trial fs used inside loop)
    sos_markers = make_lpf(LPF_MARKERS_HZ, LPF_MARKERS_ORDER, SAMPLE_RATE_HZ)

    for trial_idx in range(subject.getNumTrials()):
        trial_name = subject.getTrialName(trial_idx) or f"trial_{trial_idx:02d}"
        out_dir    = subject_out_dir / trial_name
        process_trial(subject, trial_idx, out_dir, meta, sos_markers)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {output_root / meta['subject_name']}/")


if __name__ == "__main__":
    process_subject(B3D_PATH, Path(OUTPUT_ROOT))
