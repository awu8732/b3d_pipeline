"""
process_b3d.py
==============
Single-subject B3D processing pipeline using nimblephysics.

Confirmed API against nimblephysics (Hammer2013 dataset, 3-pass B3D):
  Pass 0: KINEMATICS       -- raw IK positions
  Pass 1: LOW_PASS_FILTER  -- AddBiomechanics-filtered positions (used as IK source)
  Pass 2: DYNAMICS         -- joint moments (tau), used for ID output

Outputs per trial (OUTPUT_ROOT / <dataset_tag> / <subject_tag> / <trial_name>/):
  ik.mot           -- IK joint angles (pass 1 pos + optional extra LPF), degrees
  id_moments.sto   -- ID joint moments from pass 2 tau, N·m
  grf.mot          -- bilateral GRF + CoP, single OpenSim 18-col layout
  grf_qc.json      -- per-trial CoP-foot alignment QC (always written)
  body.json        -- subject-level body parameters

The <dataset_tag> is derived from the B3D path (e.g.
'.../test/With_Arm/Carter2023_Formatted_With_Arm/...' -> 'Carter2023_test_arm')
or set explicitly via --out-tag. Falls back to 'Other' if unparseable.

Dependencies:
  pip install nimblephysics scipy numpy

Internal modules (same directory):
  b3d_io.py       -- file writers/readers
  b3d_filters.py  -- Butterworth LPF helpers
  b3d_extract.py  -- frame-level data extraction
  b3d_grf_qc.py   -- CoP-foot alignment flagging + optional projection

CLI:
  python process_b3d.py                       # all trials
  python process_b3d.py --list                # list and exit
  python process_b3d.py -t walk_fast_1_segment_1
  python process_b3d.py -t 5 --qc-correct     # project flagged CoPs
  python process_b3d.py -t 5 --qc-threshold 0.05
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import nimblephysics as nimble
import numpy as np

from b3d_io      import write_mot, write_sto, write_body_json
from b3d_filters import make_lpf, apply_lpf
from b3d_extract import (
    read_all_frames, extract_ik, extract_id, extract_grf,
    GRF_COLUMNS,
)

# ── Extraction log ────────────────────────────────────────────────────────────
# All terminal output (stdout + stderr) is mirrored to a timestamped log file
# written to OUTPUT_ROOT/extraction_<timestamp>.log. The Tee class writes each
# line to both the original stream and the log file simultaneously.

import datetime as _dt

class _Tee:
    """Mirror writes to two streams simultaneously."""
    def __init__(self, primary, secondary):
        self._p = primary
        self._s = secondary
    def write(self, data):
        self._p.write(data)
        self._s.write(data)
        self._s.flush()
    def flush(self):
        self._p.flush()
        self._s.flush()
    def fileno(self):            # needed by os.dup2 / suppress_native_stderr
        return self._p.fileno()

def _start_extraction_log(output_root: Path, tag: str = "extraction") -> "IO[str]":
    """Open a timestamped log file and tee stdout/stderr to it."""
    output_root.mkdir(parents=True, exist_ok=True)
    ts       = _dt.datetime.now().strftime("%Y%m%d")
    log_path = output_root / f"extraction_{ts}_{tag}.log"
    log_fh   = open(log_path, "w", buffering=1)   # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"[Log] Writing extraction log to: {log_path}")
    return log_fh

from b3d_grf_qc  import (
    flag_cop_outliers, correct_cop_outliers, write_qc_sidecar,
)

# ── stderr suppression for nimble's C++ warnings ─────────────────────────────
# nimble's geometry loader warnings come from C++ writing directly to fd 2,
# which bypasses sys.stderr reassignment. We have to redirect at the OS
# file-descriptor level by dup-ing fd 2 to /dev/null. This is module-global
# (set once from CLI) and toggled by suppress_native_stderr().

SUPPRESS_NATIVE_WARNINGS = True

@contextmanager
def suppress_native_stderr():
    """
    Redirect OS-level stderr (fd 2) to /dev/null for the duration of the
    context. Captures warnings emitted by C/C++ extensions (such as nimble's
    geometry loader) that bypass Python's sys.stderr.

    No-op when SUPPRESS_NATIVE_WARNINGS is False.
    """
    if not SUPPRESS_NATIVE_WARNINGS:
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)



# ════════════════════════════════════════════════════════════════════════════
# CONFIG: edit this block; leave everything else alone
# ════════════════════════════════════════════════════════════════════════════

B3D_PATH    = "AddBiomechanicsDataset/test/With_Arm/Carter2023_Formatted_With_Arm/P010_split0/P010_split0.b3d"
OUTPUT_ROOT = "output_exp"

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

    # readSkel() emits C++ geometry-loading warnings to stderr; suppress them
    # via OS-level fd redirection (Python sys.stderr reassignment doesn't
    # catch C++ warnings).
    with suppress_native_stderr():
        skel = subject.readSkel(0)

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

    # Fix 1 + 4: getBodyScales() returns all 1.0 when nimble bakes scaling into
    # the geometry before saving (which AddBiomechanics always does). Use
    # getBodyNode(bi).getScale() instead, which reads the live per-body 3-vector
    # scale from the loaded skeleton geometry. Store as a named dict so
    # downstream consumers (b3d_grf_qc, SO pipeline) can look up by body name
    # without needing to know the skeleton body order.
    body_scales_dict: dict[str, list[float]] = {}
    try:
        for bi in range(skel.getNumBodyNodes()):
            node = skel.getBodyNode(bi)
            body_scales_dict[node.getName()] = [float(v) for v in node.getScale()]
    except Exception:
        body_scales_dict = {}

    if body_scales_dict:
        all_one = all(
            abs(v - 1.0) < 1e-6
            for scales in body_scales_dict.values()
            for v in scales
        )
        if all_one:
            print(
                "  [WARN] All body scales are 1.0. nimble may have baked "
                "scaling into geometry; per-body scale dict may be unreliable."
            )

    # Fix 2: cross-check getMassKg() against the sum of body masses in the
    # loaded skeleton. These should agree to <1%; a larger gap means the b3d
    # mass metadata and the model geometry are inconsistent.
    osim_mass_kg: float | None = None
    try:
        osim_mass_kg = sum(
            skel.getBodyNode(bi).getMass()
            for bi in range(skel.getNumBodyNodes())
        )
        mass_diff_pct = abs(osim_mass_kg - mass_kg) / max(mass_kg, 1e-6) * 100
        if mass_diff_pct > 1.0:
            print(
                f"  [WARN] Mass mismatch: getMassKg()={mass_kg:.3f} kg vs "
                f"sum of body masses={osim_mass_kg:.3f} kg "
                f"({mass_diff_pct:.1f}% difference). "
                "The written .osim may not match the declared subject mass."
            )
        else:
            print(
                f"  [OK]   Mass consistent: getMassKg()={mass_kg:.3f} kg, "
                f"model sum={osim_mass_kg:.3f} kg ({mass_diff_pct:.2f}% diff)"
            )
    except Exception:
        pass

    body_params = {
        "subject_name":   subject_name,
        "href":           href,
        "mass_kg":        mass_kg,
        "height_m":       height_m,
        "dof_names":      dof_names,
        "marker_names":   marker_names,
        "grf_bodies":     list(subject.getGroundForceBodies()),
        # Fix 4: named dict instead of a flat uninterpretable list
        "body_scales":    body_scales_dict,
    }
    if osim_mass_kg is not None:
        body_params["osim_mass_kg"] = round(osim_mass_kg, 6)

    return dict(
        subject_name     = subject_name,
        n_dofs           = n_dofs,
        dof_names        = dof_names,
        moment_names     = moment_names,
        marker_names     = marker_names,
        body_params      = body_params,
        body_scales_dict = body_scales_dict,
        skel             = skel,
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

    # Stage 2b.1: GRF QC -- always-on flagging, optional correction.
    qc_threshold_m = meta.get("qc_threshold_m", 0.03)
    qc_correct     = meta.get("qc_correct", False)

    print(f"  [Stage 2b.1] GRF QC (threshold={qc_threshold_m*100:.1f} cm, "
          f"correct={qc_correct})...")
    qc = flag_cop_outliers(
        grf            = grf,
        skel           = meta["skel"],
        ik_rad         = ik_rad,
        body_scales    = meta.get("body_scales_dict"),
        threshold_m    = qc_threshold_m,
    )
    def _cop_temporal_bars(dist_list: list, threshold_m: float,
                           t0: float, fs: float,
                           n_bins: int = 10) -> tuple[str, str, str, str]:
        """
        Two temporal bar charts over trial time from per-frame distance data.

        raw: per-bin mean CoP distance (cm), shared scale across R and L.
        adj: per-bin excess above trial mean distance, clipped at 0.
             Separate scale so small deviations stay visible.

        Returns (t0_label, t1_label, raw_bar, adj_bar).
        """
        BAR = "▁▂▃▄▅▆▇█"
        dist = np.array([v if v is not None else 0.0 for v in dist_list],
                        dtype=float) * 100.0  # -> cm
        T = len(dist)
        edges = np.linspace(0, T, n_bins + 1).astype(int)
        bin_means = [
            float(np.mean(dist[edges[b]:edges[b+1]]))
            if edges[b+1] > edges[b] else 0.0
            for b in range(n_bins)
        ]
        trial_mean = float(np.mean(dist))
        adj_bins   = [max(0.0, bm - trial_mean) for bm in bin_means]

        def _bar(vals, ceil):
            ceil = ceil or 1.0
            return "".join(
                BAR[min(int(v / ceil * (len(BAR) - 1)), len(BAR) - 1)]
                for v in vals
            )

        raw_bar = _bar(bin_means, max(bin_means) or 1.0)
        adj_bar = _bar(adj_bins,  max(adj_bins)  or 1.0)
        t_start_label = f"{t0:.2f}s"
        t_end_label   = f"{t0 + T / fs:.2f}s"
        return t_start_label, t_end_label, raw_bar, adj_bar

    for side, fr, tot, mx, mn, med, dist_key in [
        ("R",
         qc["flagged_frame_count_r"], qc["total_stance_frames_r"],
         qc["trial_max_distance_r"],  qc["trial_mean_distance_r"],
         qc["trial_median_distance_r"], "per_frame_distance_r"),
        ("L",
         qc["flagged_frame_count_l"], qc["total_stance_frames_l"],
         qc["trial_max_distance_l"],  qc["trial_mean_distance_l"],
         qc["trial_median_distance_l"], "per_frame_distance_l"),
    ]:
        t0_lbl, t1_lbl, raw_bar, adj_bar = _cop_temporal_bars(
            qc[dist_key], qc_threshold_m, t0, fs
        )
        print(f"    flagged {side}: {fr:5d} / {tot:5d} stance frames | "
              f"raw [{t0_lbl} {raw_bar} {t1_lbl}]  adj [{adj_bar}] | "
              f"(max {mx*100:.1f} cm, mean {mn*100:.1f} cm, "
              f"median {med*100:.1f} cm)")

    if qc_correct:
        grf, n_fixed = correct_cop_outliers(grf, qc, threshold_m=qc_threshold_m)
        qc["correction_applied"]    = True
        qc["n_corrected_total"]     = n_fixed
        print(f"    corrected {n_fixed} frame-side CoPs")

    write_mot(
        out_dir / "grf.mot", GRF_COLUMNS, grf, fs,
        header_name="GRF", in_degrees=False,
        t0=t0,
    )
    write_qc_sidecar(out_dir / "grf_qc.json", qc)

    # Stage 2c: body params
    write_body_json(out_dir / "body.json", meta["body_params"])


# ── Dataset tag derivation ────────────────────────────────────────────────────

# Known AddBiomechanics dataset path layout:
#   <root>/<split>/<arm_dir>/<study_dir>/<subject>/<subject>.b3d
# where:
#   split     in {"train", "test", "dev"}
#   arm_dir   in {"With_Arm", "No_Arm"}
#   study_dir is the source-paper folder, often suffixed with "_Formatted"
#             and/or "_With_Arm" / "_No_Arm". Examples seen:
#               Carter2023_Formatted_With_Arm
#               Camargo2021_Formatted_No_Arm
#               Hammer2013_Formatted
#               Falisse2017
#
# Output tag rule:
#   <study_short>_<split>_<arm_short>
# where:
#   study_short = study_dir with "_Formatted", "_With_Arm", "_No_Arm" stripped
#   arm_short   = "arm" | "noarm"
#
# Examples:
#   .../test/With_Arm/Carter2023_Formatted_With_Arm/...    -> Carter2023_test_arm
#   .../train/No_Arm/Camargo2021_Formatted_No_Arm/...      -> Camargo2021_train_noarm
#   .../dev/With_Arm/Falisse2017/...                       -> Falisse2017_dev_arm
#
# If the path doesn't conform, returns "Other".

_KNOWN_SPLITS    = {"train", "test", "dev", "val", "validation"}
_KNOWN_ARM_DIRS  = {"With_Arm": "arm", "No_Arm": "noarm"}
_STUDY_SUFFIXES  = ("_Formatted_With_Arm", "_Formatted_No_Arm",
                    "_With_Arm", "_No_Arm", "_Formatted")


def derive_dataset_tag(b3d_path: str) -> str:
    """
    Derive a short dataset tag from an AddBiomechanics-style B3D path.
    Returns 'Other' on any parse failure.
    """
    parts = Path(b3d_path).parts
    # Walk backwards from the file looking for split + arm anchor pair.
    # Expected layout: ... split / arm_dir / study_dir / subject / file.b3d
    # So if file is at index -1, study_dir at -3, arm_dir at -4, split at -5.
    if len(parts) < 5:
        return "Other"
    try:
        split   = parts[-5]
        arm_dir = parts[-4]
        study   = parts[-3]
    except IndexError:
        return "Other"

    if split not in _KNOWN_SPLITS or arm_dir not in _KNOWN_ARM_DIRS:
        return "Other"

    study_short = study
    for suf in _STUDY_SUFFIXES:
        if study_short.endswith(suf):
            study_short = study_short[: -len(suf)]
            break

    if not study_short:
        return "Other"

    return f"{study_short}_{split}_{_KNOWN_ARM_DIRS[arm_dir]}"


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
    print(f"  {n} trials:")
    for i in range(n):
        name = subject.getTrialName(i) or f"trial_{i:02d}"
        nf   = subject.getTrialLength(i)
        dt   = subject.getTrialTimestep(i)
        dur  = nf * dt if dt > 0 else 0.0
        print(f"    [{i:3d}] {name:<40s}  n={nf:5d}  dt={dt:.4f}  dur={dur:.2f}s")


def process_subject(b3d_path: str, output_root: Path,
                    trials: list | None = None,
                    list_only: bool = False,
                    qc_threshold_m: float = 0.03,
                    qc_correct: bool = False,
                    out_tag: str | None = None) -> None:
    """
    Parameters
    ----------
    b3d_path       : path to .b3d file
    output_root    : root output directory
    trials         : None = process all. Otherwise a list of trial names or
                     integer indices (mixed OK). Unknown entries raise.
    list_only      : if True, print trial list and return without processing.
    qc_threshold_m : CoP-foot distance threshold for the GRF QC step (m).
                     Frames with distance > threshold are flagged in
                     grf_qc.json, and (if qc_correct=True) projected onto
                     the foot polygon.
    qc_correct     : if True, project flagged CoPs onto the foot polygon
                     before writing grf.mot. Force vector and free torque
                     are not modified.
    out_tag        : dataset tag used as a sub-directory of output_root.
                     If None, derive_dataset_tag(b3d_path) is called; the
                     parser returns "Other" on unrecognised path layouts.
                     Final output dir is: output_root / out_tag / subject_name
    """
    print(f"\n{'='*60}")
    print(f"Loading : {b3d_path}")
    with suppress_native_stderr():
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    if list_only:
        _list_trials(subject)
        return

    meta = _load_subject_metadata(subject, b3d_path)
    meta["qc_threshold_m"] = qc_threshold_m
    meta["qc_correct"]     = qc_correct

    if out_tag is None:
        out_tag = derive_dataset_tag(b3d_path)
    print(f"Tag     : {out_tag}")
    log_fh = _start_extraction_log(Path(output_root), tag=out_tag)

    subject_out_dir = output_root / out_tag / meta["subject_name"]
    subject_out_dir.mkdir(parents=True, exist_ok=True)

    # Write shared .osim model once at the subject level.
    # Fix 3: use ID_PASS_IDX (DYNAMICS pass) instead of pass 0 (KINEMATICS).
    # AddBiomechanics finalises mass/inertia during the dynamics pass; pass 0
    # may have stale segment masses. As a safety net, compare the summed body
    # mass from each available pass against getMassKg() and prefer the closest.
    declared_mass = subject.getMassKg()
    best_pass = ID_PASS_IDX
    best_diff = float("inf")
    n_passes  = subject.getTrialNumProcessingPasses(0)
    for p in range(n_passes):
        try:
            with suppress_native_stderr():
                txt = subject.getOpensimFileText(p)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(txt)
            pass_mass = sum(
                float(el.text.strip())
                for el in root.iter("mass")
            )
            diff = abs(pass_mass - declared_mass)
            if diff < best_diff:
                best_diff  = diff
                best_pass  = p
        except Exception:
            continue

    if best_pass != ID_PASS_IDX:
        print(
            f"  [WARN] getOpensimFileText: pass {best_pass} has mass closest "
            f"to getMassKg() ({declared_mass:.3f} kg); expected pass "
            f"{ID_PASS_IDX}. Using pass {best_pass}."
        )
    else:
        print(f"  [OK]   Using getOpensimFileText(pass {best_pass}) for .osim")

    with suppress_native_stderr():
        osim_text = subject.getOpensimFileText(best_pass)
    osim_path = subject_out_dir / f"{meta['subject_name']}.osim"
    osim_path.write_text(osim_text)
    print(f"  [osim] model -> {osim_path}")

    trial_indices = _resolve_trial_selection(subject, trials)
    print(f"  Processing {len(trial_indices)} of "
          f"{subject.getNumTrials()} trials: {trial_indices}")
    print(f"  GRF QC: threshold={qc_threshold_m*100:.1f} cm, "
          f"correction={'ON' if qc_correct else 'OFF'}")

    # .trc output removed: AddBiomechanics-solved IK is the source of truth.
    total_frames = 0
    total_seconds = 0.0
    for trial_idx in trial_indices:
        trial_name = subject.getTrialName(trial_idx) or f"trial_{trial_idx:02d}"
        out_dir    = subject_out_dir / trial_name
        nf = subject.getTrialLength(trial_idx)
        dt = subject.getTrialTimestep(trial_idx)
        total_frames  += nf
        total_seconds += nf * dt if dt > 0 else nf / float(SAMPLE_RATE_HZ)
        with suppress_native_stderr():
            process_trial(subject, trial_idx, out_dir, meta)

    print(f"\n{'='*60}")
    print(f"Done. Outputs in: {subject_out_dir}/")
    print(
        f"Extracted: {len(trial_indices)} trials  |  "
        f"{total_frames:,} frames  |  "
        f"{total_seconds:.2f}s ({total_seconds/60:.2f} min)"
    )
    try:
        log_fh.flush()
        log_fh.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    except Exception:
        pass


def _parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(
        description="Process a B3D file into OpenSim-compatible outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_b3d.py
  python process_b3d.py --list
  python process_b3d.py --trial walk_fast_1_segment_1
  python process_b3d.py -t 5 -t 7
  python process_b3d.py --b3d other.b3d --out my_output -t 0
  python process_b3d.py -t 5 --qc-correct           # project flagged CoPs
  python process_b3d.py -t 5 --qc-threshold 0.05    # 5 cm tolerance
""",
    )
    p.add_argument("--b3d", default=B3D_PATH,
                   help=f"Path to .b3d file (default: {B3D_PATH})")
    p.add_argument("--out", default=OUTPUT_ROOT,
                   help=f"Output root directory (default: {OUTPUT_ROOT})")
    p.add_argument("--trial", "-t", action="append", default=None,
                   metavar="NAME_OR_INDEX",
                   help="Trial to process (name or 0-based index). Repeat "
                        "to process multiple. Omit to process all.")
    p.add_argument("--list", "-l", action="store_true", dest="list_only",
                   help="List trials in the B3D and exit.")
    p.add_argument("--qc-threshold", type=float, default=0.03,
                   metavar="METRES",
                   help="CoP-to-foot distance threshold (m). Frames with "
                        "distance above this are flagged in grf_qc.json "
                        "(default: 0.03 = 3 cm).")
    p.add_argument("--qc-correct", action="store_true",
                   help="If set, project flagged CoPs onto the foot polygon "
                        "before writing grf.mot. Force vector and free "
                        "torque are not modified.")
    p.add_argument("--out-tag", default=None, metavar="TAG",
                   help="Dataset tag used as a subfolder of --out. If "
                        "omitted, derived from the B3D path "
                        "(e.g. '.../test/With_Arm/Carter2023_Formatted_With_Arm/...' "
                        "-> 'Carter2023_test_arm'). Falls back to 'Other' "
                        "on unrecognised paths.")
    p.add_argument("--verbose-warnings", action="store_true",
                   help="Show nimble's C++ geometry-loading warnings "
                        "(missing .vtp.ply meshes, etc.). Off by default; "
                        "these warnings are cosmetic and don't affect "
                        "extraction outputs.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    SUPPRESS_NATIVE_WARNINGS = not args.verbose_warnings
    process_subject(
        b3d_path       = args.b3d,
        output_root    = Path(args.out),
        trials         = args.trial,
        list_only      = args.list_only,
        qc_threshold_m = args.qc_threshold,
        qc_correct     = args.qc_correct,
        out_tag        = args.out_tag,
    )