"""
process_b3d.py
==============
Single-subject B3D processing pipeline using nimblephysics.

Confirmed API against nimblephysics (Hammer2013 dataset, 3-pass B3D):
  Pass 0: KINEMATICS       -- raw IK positions
  Pass 1: LOW_PASS_FILTER  -- AddBiomechanics-filtered positions (used as IK source)
  Pass 2: DYNAMICS         -- joint moments (tau), used for ID output

Outputs per trial (OUTPUT_ROOT / <subject_tag> / <trial_name> /):
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

CLI:
  python process_b3d.py                         # process all trials
  python process_b3d.py --list                  # list trials and exit
  python process_b3d.py -t walk_fast_1_segment_1
  python process_b3d.py -t 5 -t 7               # indices
  python process_b3d.py --b3d other.b3d --out out2 -t 0
"""

import os
import sys
from pathlib import Path

import nimblephysics as nimble
import numpy as np

from b3d_io      import write_mot, write_sto, write_body_json
from b3d_filters import make_lpf, apply_lpf
from b3d_extract import (
    read_all_frames, extract_ik, extract_id, extract_grf,
    GRF_COLUMNS,
)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG: edit this block; leave everything else alone
# ════════════════════════════════════════════════════════════════════════════

B3D_PATH    = "AddBiomechanicsDataset/test/With_Arm/Carter2023_Formatted_With_Arm/P010_split0/P010_split0.b3d"
OUTPUT_ROOT = "output1"

# Sampling rate (Hz). The actual per-trial timestep is read from the file;
# SAMPLE_RATE_HZ is used only for filter design and as a fallback if
# getTrialTimestep() returns 0.
SAMPLE_RATE_HZ = 100

# Processing pass indices (0-based).
# Confirmed order for AddBiomechanics 3-pass B3D: 0=KIN, 1=LPF, 2=DYN.
IK_PASS_IDX = 1   # LOW_PASS_FILTER pass: source for joint positions
ID_PASS_IDX = 2   # DYNAMICS pass: source for joint moments (tau)

# Low-pass filter applied to IK positions from pass 1 (Stage 2).
# Pass 1 is already filtered by AddBiomechanics at their stored cutoff,
# so an additional filter here is typically redundant. Default None.
# Set LPF_IK_HZ to a float (e.g. 6.0) only if you need extra smoothing.
LPF_IK_HZ    = None
LPF_IK_ORDER = 4

# Frames to read per batch (reduce if RAM is tight).
READ_BATCH_SIZE = 512

# ════════════════════════════════════════════════════════════════════════════


# ── IK post-processing ────────────────────────────────────────────────────────

def _append_beta_coords(ik_out: np.ndarray, dof_names: list) -> tuple[np.ndarray, list]:
    """
    Append patellofemoral beta coordinates required by the LaiArnold model.

    The CoordinateCouplerConstraint uses a LinearFunction with coefficients
    [1, 0], i.e. beta = 1 * knee_angle + 0, evaluated by OpenSim in RADIANS
    (internal units). In the .mot, standard rotational coords like
    'knee_angle_r' are auto-converted deg->rad on load (because inDegrees=yes
    and they are typed as Rotational). The 'knee_angle_*_beta' coords in the
    LaiArnold/Rajagopal model are NOT auto-converted by the .mot loader, so
    they must be written in radians even when the rest of the file is degrees.
    Writing beta in degrees produces a ~57x overshoot at the patella, pulling
    vastus/rectus attachments off the femur (classic 'distended vasti' bug).

    This function is called AFTER _rad_to_deg_ik, so ik_out already has the
    knee_angle_* columns in degrees; we therefore convert them back to radians
    when populating the beta columns.

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
            # ik_out[:, src_idx] is in degrees; beta must be written in radians
            beta_data[beta_name] = np.radians(ik_out[:, src_idx])

    if not beta_data:
        return ik_out, dof_names

    extra = np.column_stack(list(beta_data.values()))
    ik_out = np.hstack([ik_out, extra])
    print(f"    Appended beta coords in radians: {list(beta_data.keys())}")
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
) -> None:
    """
    Run the full extraction + write pipeline for a single trial.

    Stages:
      1. IK         -> ik.mot
      2a. ID        -> id_moments.sto
      2b. GRF       -> grf.mot
      2c. Body      -> body.json
    """
    trial_name = subject.getTrialName(trial_idx) or f"trial_{trial_idx:02d}"
    n_frames   = subject.getTrialLength(trial_idx)
    timestep   = subject.getTrialTimestep(trial_idx)
    fs         = (1.0 / timestep) if timestep > 0 else float(SAMPLE_RATE_HZ)

    # Fix #11: recover this trial's absolute start time if it's a split piece
    # of a longer original recording. nimble exposes getTrialSplitIndex(),
    # which is the 0-based ordinal of this piece within the original trial.
    # Assuming pieces are contiguous and equal length, t0 = split * dt * n.
    # (If the API isn't available in this build, fall back to t0 = 0.)
    try:
        split_idx = subject.getTrialSplitIndex(trial_idx)
    except Exception:
        split_idx = 0
    t0 = float(split_idx) * n_frames * timestep if timestep > 0 else 0.0

    print(f"\n{'-'*50}")
    print(f"  Trial {trial_idx}: {trial_name}")
    print(f"  Frames: {n_frames}  fs: {fs:.4f} Hz  "
          f"({n_frames / fs:.2f} s, t0={t0:.3f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Reading frames...", end=" ", flush=True)
    frames = read_all_frames(subject, trial_idx, n_frames, READ_BATCH_SIZE)
    print(f"{len(frames)} loaded.")

    n_dofs       = meta["n_dofs"]
    dof_names    = meta["dof_names"]
    moment_names = meta["moment_names"]

    # Stage 1 (markers -> .trc) intentionally skipped: downstream consumers
    # (IK is already solved by AddBiomechanics, so ik.mot is the source of
    # truth; markers are not re-used).

    # Stage 1: IK
    print(f"  [Stage 1] IK (pass {IK_PASS_IDX}: "
          f"{'LOW_PASS_FILTER' if IK_PASS_IDX == 1 else 'other'})...")
    ik_rad = extract_ik(frames, n_dofs, IK_PASS_IDX)
    if LPF_IK_HZ is not None:
        sos_ik = make_lpf(LPF_IK_HZ, LPF_IK_ORDER, fs)
        ik_rad = apply_lpf(ik_rad, sos_ik)
        print(f"    Additional LPF applied: {LPF_IK_HZ} Hz order {LPF_IK_ORDER}")
    else:
        print("    No additional LPF (pass already filtered by "
              "AddBiomechanics; used as-is).")

    ik_out                 = _rad_to_deg_ik(ik_rad, dof_names)
    ik_out, aug_dof_names  = _append_beta_coords(ik_out, dof_names)

    write_mot(
        out_dir / "ik.mot", aug_dof_names,
        ik_out, fs,
        header_name="Coordinates", in_degrees=True,
        t0=t0,
    )

    # Stage 2a: ID moments
    print("  [Stage 2a] ID moments (pass 2: DYNAMICS)...")
    tau = extract_id(frames, n_dofs, ID_PASS_IDX)
    write_sto(
        out_dir / "id_moments.sto", moment_names, tau, fs,
        header_name="InverseDynamics",
        t0=t0,
    )

    # Stage 2b: GRF
    print("  [Stage 2b] GRF (from dynamics pass)...")
    grf = extract_grf(frames, pass_idx=ID_PASS_IDX)
    write_mot(
        out_dir / "grf.mot", GRF_COLUMNS, grf, fs,
        header_name="GRF", in_degrees=False,
        t0=t0,
    )

    # Stage 2c: body params
    write_body_json(out_dir / "body.json", meta["body_params"])


# ── Top-level entry point ─────────────────────────────────────────────────────

def _resolve_trial_selection(subject, trials: list | None) -> list[int]:
    """
    Resolve a list of trial selectors (names, indices, or None=all) into a
    de-duplicated, sorted list of valid trial indices. Raises on unknown names
    or out-of-range indices.
    """
    n = subject.getNumTrials()
    all_names = [subject.getTrialName(i) or f"trial_{i:02d}" for i in range(n)]

    if not trials:
        return list(range(n))

    resolved: list[int] = []
    for sel in trials:
        # Try integer index first
        try:
            idx = int(sel)
        except (TypeError, ValueError):
            idx = None

        if idx is not None:
            if not (0 <= idx < n):
                raise ValueError(
                    f"Trial index {idx} out of range (0..{n - 1})."
                )
            resolved.append(idx)
        else:
            if sel not in all_names:
                raise ValueError(
                    f"Trial name '{sel}' not found. Use --list to see "
                    f"available trials."
                )
            resolved.append(all_names.index(sel))

    return sorted(set(resolved))


def _list_trials(subject) -> None:
    """Print one line per trial: index, name, frames, dt, duration."""
    n = subject.getNumTrials()
    print(f"  {n} trials in {Path.cwd()}:")
    for i in range(n):
        name = subject.getTrialName(i) or f"trial_{i:02d}"
        nf   = subject.getTrialLength(i)
        dt   = subject.getTrialTimestep(i)
        dur  = nf * dt if dt > 0 else 0.0
        print(f"    [{i:3d}] {name:<40s}  n={nf:5d}  dt={dt:.4f}  dur={dur:.2f}s")


def process_subject(b3d_path: str, output_root: Path,
                    trials: list | None = None,
                    list_only: bool = False) -> None:
    """
    Parameters
    ----------
    b3d_path    : path to .b3d file
    output_root : root output directory
    trials      : None = process all. Otherwise a list of trial names or
                  integer indices (mixed OK). Unknown entries raise.
    list_only   : if True, print trial list and return without processing.
    """
    print(f"\n{'='*60}")
    print(f"Loading : {b3d_path}")
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    if list_only:
        _list_trials(subject)
        return

    meta = _load_subject_metadata(subject, b3d_path)

    subject_out_dir = output_root / meta["subject_name"]
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    # Write shared .osim model once at the subject level
    osim_text = subject.getOpensimFileText(0)
    osim_path = subject_out_dir / f"{meta['subject_name']}.osim"
    osim_path.write_text(osim_text)
    print(f"  [osim] model -> {osim_path}")

    trial_indices = _resolve_trial_selection(subject, trials)
    print(f"  Processing {len(trial_indices)} of "
          f"{subject.getNumTrials()} trials: {trial_indices}")

    # .trc output removed: AddBiomechanics-solved IK is the source of truth.
    for trial_idx in trial_indices:
        trial_name = subject.getTrialName(trial_idx) or f"trial_{trial_idx:02d}"
        out_dir    = subject_out_dir / trial_name
        process_trial(subject, trial_idx, out_dir, meta)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {output_root / meta['subject_name']}/")


def _parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(
        description="Process a B3D file into OpenSim-compatible outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all trials (default paths from CONFIG block):
  python process_b3d.py

  # List trials without processing:
  python process_b3d.py --list

  # Process one trial by name:
  python process_b3d.py --trial walk_fast_1_segment_1

  # Process multiple trials (names or indices, mixed):
  python process_b3d.py --trial walk_fast_1_segment_1 --trial 5 --trial Static_1_segment_0

  # Override B3D and output paths:
  python process_b3d.py --b3d path/to/other.b3d --out my_output --trial 5
""",
    )
    p.add_argument("--b3d", default=B3D_PATH,
                   help=f"Path to .b3d file (default from CONFIG: {B3D_PATH})")
    p.add_argument("--out", default=OUTPUT_ROOT,
                   help=f"Output root directory (default: {OUTPUT_ROOT})")
    p.add_argument("--trial", "-t", action="append", default=None,
                   metavar="NAME_OR_INDEX",
                   help="Trial to process (name or 0-based index). Repeat "
                        "to process multiple. Omit to process all.")
    p.add_argument("--list", "-l", action="store_true", dest="list_only",
                   help="List trials in the B3D and exit without processing.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    process_subject(
        b3d_path    = args.b3d,
        output_root = Path(args.out),
        trials      = args.trial,
        list_only   = args.list_only,
    )