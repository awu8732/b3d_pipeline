"""
diagnose_b3d.py
===============
Diagnostic script to inspect what nimblephysics exposes on Frame and
ProcessingPass objects, and compare raw vs processed GRF data.

Usage:
    python diagnose_b3d.py <path_to_b3d_file>

Example:
    python diagnose_b3d.py AddBiomechanicsDataset/train/With_Arm/Carter2023_Formatted_With_Arm/P003_split0/P003_split0.b3d
"""

import sys
import json
from pathlib import Path
from pprint import pprint

import numpy as np

try:
    import nimblephysics as nimble
except ImportError:
    print("ERROR: nimblephysics not installed. pip install nimblephysics")
    sys.exit(1)


def inspect_attributes(obj, label: str, skip_private: bool = True):
    """Print all attributes/methods on an object, grouped by type."""
    attrs = sorted(dir(obj))
    if skip_private:
        attrs = [a for a in attrs if not a.startswith("_")]

    properties = []
    methods = []
    for a in attrs:
        try:
            val = getattr(obj, a)
            if callable(val):
                methods.append(a)
            else:
                properties.append((a, type(val).__name__, repr(val)[:120]))
        except Exception as e:
            properties.append((a, "ERROR", str(e)[:120]))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\n  Properties ({len(properties)}):")
    for name, typ, val in properties:
        print(f"    {name:45s} [{typ:15s}] = {val}")
    print(f"\n  Methods ({len(methods)}):")
    for m in methods:
        print(f"    {m}()")
    print()


def compare_grf_sources(frame, pass_idx: int):
    """Compare raw force plate data vs processing pass ground contact data."""
    print(f"\n{'='*70}")
    print(f"  GRF source comparison (frame)")
    print(f"{'='*70}")

    # --- Raw force plate data (what you currently use) ---
    print("\n  [Raw force plate data]")
    for attr in ["rawForcePlateForces", "rawForcePlateCenterOfPressures",
                 "rawForcePlateTorques"]:
        try:
            val = getattr(frame, attr)
            print(f"    {attr}:")
            for i, v in enumerate(val):
                arr = np.array(v)
                print(f"      plate[{i}]: {arr}")
        except Exception as e:
            print(f"    {attr}: NOT AVAILABLE ({e})")

    # --- Processing pass ground contact data ---
    pp = frame.processingPasses[pass_idx]
    print(f"\n  [Processing pass {pass_idx} ground contact data]")

    # Try every plausible attribute name for processed GRFs
    grf_candidates = [
        "groundContactForce", "groundContactForces",
        "groundContactCenterOfPressure", "groundContactCentersOfPressure",
        "groundContactTorque", "groundContactTorques",
        "groundContactWrench", "groundContactWrenches",
        "groundForce", "groundForces",
        "comForce", "externalForce", "externalForces",
        "contactForce", "contactForces",
        "force", "forces",
        "grfForce", "grfForces",
    ]
    found_any = False
    for attr in grf_candidates:
        if hasattr(pp, attr):
            try:
                val = getattr(pp, attr)
                arr = np.array(val)
                print(f"    {attr}: shape={arr.shape}  values={arr.ravel()[:12]}")
                found_any = True
            except Exception as e:
                print(f"    {attr}: exists but cannot read ({e})")
                found_any = True

    if not found_any:
        print("    No known GRF attributes found on processing pass.")
        print("    Full attribute list printed above; look for anything GRF-related.")


def check_tau_consistency(frames, pass_idx: int, n_check: int = 5):
    """Print tau (joint moments) from the processing pass for a few frames."""
    print(f"\n{'='*70}")
    print(f"  Tau (joint moments) from pass {pass_idx}, first {n_check} frames")
    print(f"{'='*70}")
    for i in range(min(n_check, len(frames))):
        tau = np.array(frames[i].processingPasses[pass_idx].tau)
        print(f"    frame {i}: min={tau.min():.2f}  max={tau.max():.2f}  "
              f"first 6: {tau[:6]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_b3d.py <path_to_b3d_file>")
        sys.exit(1)

    b3d_path = sys.argv[1]
    print(f"Loading: {b3d_path}")
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # --- Subject-level info ---
    n_trials = subject.getNumTrials()
    n_dofs = subject.getNumDofs()
    print(f"Trials: {n_trials}   DOFs: {n_dofs}")
    print(f"Mass: {subject.getMassKg():.1f} kg   Height: {subject.getHeightM():.3f} m")

    # --- Processing pass info ---
    trial_idx = 0
    n_passes = subject.getTrialNumProcessingPasses(trial_idx)
    print(f"\nTrial {trial_idx}: {subject.getTrialName(trial_idx)}")
    print(f"  Frames: {subject.getTrialLength(trial_idx)}")
    print(f"  Processing passes: {n_passes}")

    for p in range(n_passes):
        pass_type = subject.getProcessingPassType(p)
        print(f"  Pass {p}: {pass_type}")

    # --- Ground force bodies ---
    grf_bodies = list(subject.getGroundForceBodies())
    print(f"\n  Ground force bodies: {grf_bodies}")

    # --- Read a small batch of frames for inspection ---
    print("\nReading 10 frames for inspection...")
    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=10,       # skip frame 0 (often zeroed out)
        numFramesToRead=10,
        includeSensorData=True,
        includeProcessingPasses=True,
    )
    frame = frames[0]

    # --- Inspect Frame object ---
    inspect_attributes(frame, "Frame object (frames[0])")

    # --- Inspect each processing pass ---
    for p in range(n_passes):
        pp = frame.processingPasses[p]
        inspect_attributes(pp, f"ProcessingPass[{p}]")

    # --- Compare GRF sources ---
    dynamics_pass = n_passes - 1  # typically the last pass
    compare_grf_sources(frame, dynamics_pass)

    # --- Check tau values ---
    check_tau_consistency(frames, dynamics_pass)

    # --- Compare raw GRF magnitude vs tau magnitude ---
    print(f"\n{'='*70}")
    print(f"  Quick consistency check: raw GRF vertical vs body weight")
    print(f"{'='*70}")
    mass = subject.getMassKg()
    bw = mass * 9.81
    for i in range(min(5, len(frames))):
        rf = np.array(frames[i].rawForcePlateForces)
        total_fy = sum(rf[plate][1] for plate in range(rf.shape[0]))
        print(f"    frame {10+i}: total vertical GRF = {total_fy:.1f} N  "
              f"({total_fy/bw:.2f} BW)  [expected ~1.0 BW during stance]")

    print(f"\n{'='*70}")
    print("  DONE. Check the ProcessingPass attributes above for any")
    print("  ground-contact or force-related fields that differ from")
    print("  rawForcePlateForces. If none exist, you will need to add")
    print("  reserve actuators to absorb the residuals.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()