"""
diagnose_p003.py
================
Targeted diagnostic for P003 walk_fast to:
  1. Confirm pass indices and types for this specific trial
  2. Compare raw vs processed GRFs with left/right verification
  3. Check pass 2 tau magnitudes (are they reasonable?)
  4. Verify which foot is which in the 6-vector packing
  5. Compare what process_b3d wrote vs what the pass actually contains

Usage:
    python diagnose_p003.py <b3d_path> [trial_name_substring]

Example:
    python diagnose_p003.py path/to/P003_split0.b3d walk_fast
"""

import sys
import numpy as np

try:
    import nimblephysics as nimble
except ImportError:
    print("ERROR: nimblephysics not installed")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_p003.py <b3d_path> [trial_name_substring]")
        sys.exit(1)

    b3d_path = sys.argv[1]
    trial_filter = sys.argv[2].lower() if len(sys.argv) > 2 else "walk_fast"

    print(f"Loading: {b3d_path}")
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    mass = subject.getMassKg()
    bw = mass * 9.81
    print(f"Mass: {mass:.1f} kg  BW: {bw:.1f} N")
    print(f"Trials: {subject.getNumTrials()}  DOFs: {subject.getNumDofs()}")
    print(f"Ground force bodies: {list(subject.getGroundForceBodies())}")

    # Find the target trial
    target_trial = None
    for i in range(subject.getNumTrials()):
        name = subject.getTrialName(i)
        if trial_filter in name.lower():
            target_trial = i
            print(f"\nFound trial {i}: '{name}'")
            break

    if target_trial is None:
        print(f"\nNo trial matching '{trial_filter}'. Available trials:")
        for i in range(subject.getNumTrials()):
            print(f"  {i}: {subject.getTrialName(i)}")
        sys.exit(1)

    n_frames = subject.getTrialLength(target_trial)
    n_passes = subject.getTrialNumProcessingPasses(target_trial)
    timestep = subject.getTrialTimestep(target_trial)
    print(f"Frames: {n_frames}  Passes: {n_passes}  dt: {timestep:.6f}s  "
          f"fs: {1/timestep:.1f} Hz")

    for p in range(n_passes):
        print(f"  Pass {p}: {subject.getProcessingPassType(p)}")

    # ── Read frames 1-20 (skip frame 0 which is often zeroed) ──
    print(f"\nReading frames 1-20...")
    frames = subject.readFrames(
        trial=target_trial, startFrame=1, numFramesToRead=20,
        includeSensorData=True, includeProcessingPasses=True,
    )

    # ── 1. Pass type and tau for each pass ──
    print(f"\n{'='*70}")
    print("  1. TAU COMPARISON ACROSS ALL PASSES (frame 5)")
    print(f"{'='*70}")
    f = frames[5]
    n_dofs = subject.getNumDofs()

    # Get DOF names from skeleton
    import os
    with open(os.devnull, "w") as dn:
        old, sys.stderr = sys.stderr, dn
        try:
            skel = subject.readSkel(0)
        finally:
            sys.stderr = old
    dof_names = [dof.getName() for dof in skel.getDofs()]

    for p in range(n_passes):
        pp = f.processingPasses[p]
        tau = np.array(pp.tau)
        print(f"\n  Pass {p} ({pp.type}):")
        print(f"    linearResidual:  {pp.linearResidual:.2f} N")
        print(f"    angularResidual: {pp.angularResidual:.2f} N·m")
        print(f"    tau range: [{tau.min():.2f}, {tau.max():.2f}]")
        # Print pelvis tau specifically
        for di, name in enumerate(dof_names[:6]):
            print(f"    tau[{name}] = {tau[di]:.4f}")

    # ── 2. GRF: raw vs processed, with left/right verification ──
    print(f"\n{'='*70}")
    print("  2. GRF RAW vs PROCESSED (frames 3-8)")
    print(f"{'='*70}")

    for fi in range(3, 9):
        frame = frames[fi]
        t = (fi + 1) * timestep  # +1 because we started at frame 1

        # Raw
        raw_f = [np.array(x) for x in frame.rawForcePlateForces]
        raw_c = [np.array(x) for x in frame.rawForcePlateCenterOfPressures]

        # Processed (pass 2)
        pp2 = frame.processingPasses[n_passes - 1]  # last pass = dynamics
        proc_f = np.array(pp2.groundContactForce)
        proc_c = np.array(pp2.groundContactCenterOfPressure)
        contact = np.array(pp2.contact)

        print(f"\n  Frame {fi+1} (t={t:.4f}s)  contact={contact}")
        print(f"    Raw plate[0] force:  {raw_f[0]}  CoP_x={raw_c[0][0]:.4f}")
        print(f"    Raw plate[1] force:  {raw_f[1]}  CoP_x={raw_c[1][0]:.4f}")
        print(f"    Proc 6vec[0:3] force: {proc_f[0:3]}  CoP_x={proc_c[0]:.4f}")
        print(f"    Proc 6vec[3:6] force: {proc_f[3:6]}  CoP_x={proc_c[3]:.4f}")
        total_raw = sum(r[1] for r in raw_f)
        total_proc = proc_f[1] + proc_f[4]
        print(f"    Total vertical: raw={total_raw:.1f} N  proc={total_proc:.1f} N  "
              f"({total_proc/bw:.2f} BW)")

    # ── 3. Verify which 6vec half is which foot ──
    print(f"\n{'='*70}")
    print("  3. LEFT/RIGHT FOOT IDENTIFICATION")
    print(f"     Looking for single-support frames to identify which")
    print(f"     half of the 6-vector is which foot...")
    print(f"{'='*70}")

    # Read more frames to find single-support
    more_frames = subject.readFrames(
        trial=target_trial, startFrame=0, numFramesToRead=min(500, n_frames),
        includeSensorData=True, includeProcessingPasses=True,
    )
    dyn_pass = n_passes - 1

    found_examples = 0
    for fi, frame in enumerate(more_frames):
        pp = frame.processingPasses[dyn_pass]
        contact = np.array(pp.contact)
        proc_f = np.array(pp.groundContactForce)
        proc_c = np.array(pp.groundContactCenterOfPressure)
        vert_a = proc_f[1]  # vertical component of 6vec[0:3]
        vert_b = proc_f[4]  # vertical component of 6vec[3:6]

        # Single support: one side has significant force, other near zero
        if vert_a > 50 and vert_b < 5:
            cop_x = proc_c[0]
            print(f"\n  Frame {fi}: ONLY 6vec[0:3] active")
            print(f"    Force: {proc_f[0:3]}")
            print(f"    CoP:   {proc_c[0:3]}  (x={cop_x:.4f})")
            print(f"    contact flags: {contact}")
            print(f"    -> If CoP_x is on the LEFT side of the body, "
                  f"6vec[0:3] = LEFT foot")
            found_examples += 1
        elif vert_b > 50 and vert_a < 5:
            cop_x = proc_c[3]
            print(f"\n  Frame {fi}: ONLY 6vec[3:6] active")
            print(f"    Force: {proc_f[3:6]}")
            print(f"    CoP:   {proc_c[3:6]}  (x={cop_x:.4f})")
            print(f"    contact flags: {contact}")
            print(f"    -> If CoP_x is on the LEFT side of the body, "
                  f"6vec[3:6] = LEFT foot")
            found_examples += 1

        if found_examples >= 4:
            break

    if found_examples == 0:
        print("  No clear single-support frames found in first 500 frames.")
        print("  Showing CoP positions for both halves at frame 50:")
        pp = more_frames[50].processingPasses[dyn_pass]
        proc_c = np.array(pp.groundContactCenterOfPressure)
        print(f"    6vec[0:3] CoP: {proc_c[0:3]}")
        print(f"    6vec[3:6] CoP: {proc_c[3:6]}")

    # ── 4. Check: does the IK have coordinates the model doesn't? ──
    print(f"\n{'='*70}")
    print("  4. DOF NAMES FROM SKELETON")
    print(f"{'='*70}")
    print(f"  DOF count: {n_dofs}")
    for i, name in enumerate(dof_names):
        print(f"    {i:2d}: {name}")

    # ── 5. What your ik_filtered.mot columns are ──
    print(f"\n{'='*70}")
    print("  5. IK COLUMN CHECK")
    print(f"     Checking if ik_filtered.mot has columns not in the model...")
    print(f"{'='*70}")
    ik_cols_expected = dof_names.copy()
    # The beta coords are appended by process_b3d
    print(f"  Model DOFs: {len(dof_names)}")
    print(f"  ik_filtered.mot has 39 data columns (40 - time)")
    print(f"  Model has {len(dof_names)} DOFs")
    extra = 39 - len(dof_names)
    if extra > 0:
        print(f"  -> {extra} extra columns in IK file (likely beta coords)")
        print(f"     These must be locked or the model must have matching coords")
    elif extra < 0:
        print(f"  -> MODEL has {-extra} more DOFs than the IK file!")
        print(f"     Missing DOFs will default to 0, which may cause huge residuals")

    # ── 6. Check pelvis position in IK vs what pass 2 has ──
    print(f"\n{'='*70}")
    print("  6. IK POSITION COMPARISON: pass 2 pos vs what was written")
    print(f"{'='*70}")
    for fi in [1, 5, 10]:
        pp = frames[fi].processingPasses[dyn_pass]
        pos = np.array(pp.pos)
        print(f"\n  Frame {fi+1}:")
        for di in range(min(7, len(dof_names))):
            val_rad = pos[di]
            is_trans = "tx" in dof_names[di] or "ty" in dof_names[di] or "tz" in dof_names[di]
            if is_trans:
                print(f"    {dof_names[di]:20s} = {val_rad:.6f} m")
            else:
                print(f"    {dof_names[di]:20s} = {val_rad:.6f} rad  "
                      f"({np.degrees(val_rad):.4f} deg)")


if __name__ == "__main__":
    main()